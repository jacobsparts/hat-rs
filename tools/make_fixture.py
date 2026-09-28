#!/usr/bin/env python3
"""Write the golden fixtures hat-rs's tests check against.

The engine is checked against the PUBLISHED network (tools/hat_arch.py, copied
verbatim from XPixelGroup/HAT), not against its own reading of it: a CPU and a GPU
backend that agree with each other prove only that they share a mistake. This
script builds that network with a released configuration, loads the released
weights, runs a seeded input through it, and writes the input and the output as
raw little-endian f32 so the Rust side needs no numpy at test time.

    python3 tools/make_fixture.py --model ../models/HAT-S_SRx4.pth --config s \
        --scale 4 --h 48 --w 48 --seed 1 --out tests/data/hats_x4_48x48.bin

THE INPUT SIZE MUST BE A MULTIPLE OF THE WINDOW (16). The published `HAT.forward`
has no padding: it calls `window_partition`, which reshapes h into (h/16, 16), so
a 37x29 input raises inside the reference before the engine is ever reached. The
engine pads to a window multiple itself (see src/plan.rs) and the reference is
fed the already-padded input, which is what makes the two comparable.

    magic "HATF" (4 bytes) | u32 version | u32 h | u32 w | u32 c | u32 scale
    u32 win | u32 flags | u32 reserved | f32 input[h*w*c] | f32 expected[oh*ow*c]

`--dump` also writes every intermediate activation as a `.pt` file, which is how a
divergence is located by STAGE rather than by bisecting the output image. The dump
names are the reference's own module paths plus a suffix for the tensors the
forward computes inline (`_attn`, `_conv_x`, ...); `--dump-list` prints them
without running anything.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timm_shim  # noqa: E402

timm_shim.install()

from hat_arch import HAT  # noqa: E402

# The released configurations, from the `network_g` blocks of
# options/test/HAT-S_SRx4.yml, HAT_SRx4.yml and HAT-L_SRx4_ImageNet-pretrain.yml.
CONFIGS = {
    "s": dict(depths=[6] * 6, embed_dim=144, num_heads=[6] * 6, mlp_ratio=2,
              compress_ratio=24, squeeze_factor=24),
    "m": dict(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
              compress_ratio=3, squeeze_factor=30),
    "l": dict(depths=[6] * 12, embed_dim=180, num_heads=[6] * 12, mlp_ratio=2,
              compress_ratio=3, squeeze_factor=30),
}
# Every released HAT is window 16, overlap 0.5, conv_scale 0.01, resi 1conv,
# pixelshuffle, img_range 1.0.
COMMON = dict(in_chans=3, img_size=64, window_size=16, overlap_ratio=0.5,
              conv_scale=0.01, img_range=1., upsampler="pixelshuffle",
              resi_connection="1conv")
MAGIC = b"HATF"
VERSION = 1
FLAG_AUX = 1  # no released HAT head produces a second image; the bit is reserved


def build(config, scale):
    return HAT(upscale=scale, **CONFIGS[config], **COMMON)


def load_state(path):
    """The released checkpoints hold `params` and `params_ema`, and they differ by
    up to 6e-4 - they are not the same weights. BasicSR's test configs all name
    `param_key_g: 'params_ema'`, so that is the set the reference uses and the set
    the engine must be converted from; taking `params` would be a different (and
    sometimes non-finite, as upstream's README warns) network."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params"):
        if isinstance(obj, dict) and k in obj:
            return obj[k], k
    return obj, "raw"


class Dumper:
    """Records activations by hooking the reference, so the engine can be diffed
    stage by stage.

    Module hooks are not enough on their own: the interesting tensors inside a
    block (`conv_x`, the attention scores) are computed inline in `forward` and
    belong to no module. Those are captured by monkey-patching the two functions
    that produce them for the duration of the run, which leaves `hat_arch.py`
    itself untouched.
    """

    def __init__(self, model, enabled):
        self.store = {}
        self.enabled = enabled
        if not enabled:
            return
        self.handles = []
        for name, mod in model.named_modules():
            if name and len(list(mod.children())) == 0 and not list(mod.parameters(recurse=False)):
                continue  # nothing to see at a leaf with no weights of its own
            if name:
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(mod, inp, out):
            if isinstance(out, torch.Tensor):
                self.store[name] = out.detach().clone()
        return fn

    def close(self):
        for h in getattr(self, "handles", []):
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="released .pth checkpoint (or a state dict)")
    ap.add_argument("--config", default="s", choices=sorted(CONFIGS))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--w", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="fixture to write")
    ap.add_argument("--dump", help="directory for per-stage .pt files")
    ap.add_argument("--save-weights", help="write a random-weight state dict and exit")
    ap.add_argument("--dump-list", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = build(args.config, args.scale)
    if args.model:
        sd, which = load_state(args.model)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"checkpoint {args.model}: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys", file=sys.stderr)
            for k in missing[:8]:
                print(f"  missing:    {k}", file=sys.stderr)
            for k in unexpected[:8]:
                print(f"  unexpected: {k}", file=sys.stderr)
            if missing or unexpected:
                return 1
        print(f"loaded params from `{which}` of {args.model}")
    model.eval()

    if args.dump_list:
        for k in sorted(model.state_dict()):
            print(k)
        return 0

    if args.save_weights:
        torch.save({"params_ema": model.state_dict()}, args.save_weights)
        print(f"wrote {args.save_weights}")
        return 0

    if args.out is None:
        print("--out is required unless --dump-list/--save-weights", file=sys.stderr)
        return 1
    win = COMMON["window_size"]
    if args.h % win or args.w % win:
        print(f"the reference needs h and w to be multiples of the window ({win}): "
              f"{args.h}x{args.w} raises inside window_partition", file=sys.stderr)
        return 1

    gen = torch.Generator().manual_seed(args.seed)
    x = torch.rand(1, 3, args.h, args.w, generator=gen)
    dump = Dumper(model, bool(args.dump))
    with torch.no_grad():
        y = model(x)
    dump.close()

    if args.dump:
        d = Path(args.dump)
        d.mkdir(parents=True, exist_ok=True)
        for name, t in dump.store.items():
            torch.save(t, d / (name.replace(".", "_") + ".pt"))
        torch.save(x, d / "input.pt")
        torch.save(y, d / "output.pt")
        print(f"dumped {len(dump.store) + 2} tensors to {d}")

    header = struct.pack("<4sIIIIIIII", MAGIC, VERSION, args.h, args.w, 3,
                         args.scale, win, 0, 0)
    with open(args.out, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(x.numpy(), dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(y.numpy(), dtype="<f4").tobytes())
    print(f"wrote {args.out}: {args.h}x{args.w} -> {args.h * args.scale}x{args.w * args.scale}, "
          f"{Path(args.out).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
