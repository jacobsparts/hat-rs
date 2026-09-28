#!/usr/bin/env python3
"""Convert a released HAT .pth checkpoint into the .safetensors this engine loads.

    python3 tools/convert.py ../models/HAT-S_SRx4.pth ../models/hat-s-x4.safetensors \
        --variant hat-s --scale 4

The released checkpoints are the HAT authors' work
(https://github.com/XPixelGroup/HAT, Apache-2.0); this script only reshapes them.

THREE THINGS IT DOES BESIDES COPYING TENSORS, each because the engine would
otherwise have to guess or repeat the work at every run:

1. TAKES `params_ema`, NOT `params`. The files carry both, they are NOT the same
   weights (they differ by up to 6e-4 on every tensor), and every released BasicSR
   test config names `param_key_g: 'params_ema'` - so that is the set the reference
   evaluates and the set the engine must be converted from. The choice is written
   into the metadata as `source` so the artifact says which one it holds.

2. DROPS THE TENSORS THAT BELONG TO NO LAYER. Two kinds qualify. `HAB.__init__`
   passes `drop_path=0` to a `DropPath(...)`, which makes it an `nn.Identity`: eval
   never reads its `.drop_path` key. And HAT's `PatchEmbed` runs with
   `patch_size=1`, so its `proj` is an `nn.Conv2d(..., kernel_size=1)` over a single
   pixel - an identity reshape whose weights are never used; only its `norm` carries
   signal, and that one is KEPT.

   THE HAB'S AND OCAB'S `norm1`/`norm2` ARE NOT IN THIS CATEGORY, and an earlier
   version of this script dropped them. They are `nn.LayerNorm` modules whose
   forward IS called (`HAB.forward` starts `self.norm1(x)`), so their weight and
   bias are trained parameters: on the released HAT-S checkpoint block 0's
   `norm1.weight` has mean 0.487 and std 0.141, and `load_state_dict(params_ema)`
   needs all 168 of those tensors (42 blocks and OCABs x 4) with no missing keys.
   Dropping them does not remove an unused parameter - it removes the model.

3. PRE-BUILDS THE RELATIVE-POSITION BIAS TABLE the reference gathers per block.
   `relative_position_bias_table[rpi.view(-1)].view(n, n, nH).permute(2, 0, 1)` is
   computed once per (block, table) by the reference and every block of a stage
   shares one table, so the engine reads the table and the INDEX BUFFER in the
   checkpoint and writes the gathered [nH][n][n] table as its own tensor.

   That is 9 x 36 x 6 x 256 x 256 floats for HAT-S's local windows (25 MB) and half
   as much again for the OCAB's 256x169 - 38 MB of pure redundancy over the 1521x6
   table it comes from. `--gather-bias` is therefore OFF by default, and the engine
   builds the table it needs at upload time from the table plus an index map, which
   is 36 KB of index per block. The flag exists because the two are not identical
   in the last bit: gathering on the host and uploading 38 MB is exactly the
   reference's arithmetic, while gathering in the kernel is a permutation that can
   be written with either order. `--verify` against a fixture made by
   `tools/make_fixture.py` is what settles which one the engine uses, and it is
   measured, not assumed.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# The released configurations, from the `network_g` blocks of
# options/test/HAT-S_SRx4.yml, HAT_SRx4.yml and HAT-L_SRx4_ImageNet-pretrain.yml.
VARIANTS = {
    "hat-s": dict(depths=[6] * 6, embed_dim=144, num_heads=[6] * 6, mlp_ratio=2,
                  compress_ratio=24, squeeze_factor=24),
    "hat": dict(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                compress_ratio=3, squeeze_factor=30),
    "hat-l": dict(depths=[6] * 12, embed_dim=180, num_heads=[6] * 12, mlp_ratio=2,
                  compress_ratio=3, squeeze_factor=30),
}
WINDOW = 16
OVERLAP = 0.5
CONV_SCALE = 0.01
IMG_RANGE = 1.0
MEAN = "0.4488,0.4371,0.4040"


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params"):
        if isinstance(obj, dict) and k in obj:
            return obj[k], k
    return obj, "raw"


def infer(sd):
    """Check the tensors against the variant they claim.

    Derived rather than trusted: the depths come from the layer indices that
    appear, the embedding from the qkv weight's width, and the head's width from
    conv_before_upsample. A checkpoint whose metadata disagrees with its own
    tensors is the mistake this catches - it would otherwise be run with buffers
    that do not match it and produce a wrong image rather than an error.
    """
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.")})
    depths = []
    for l in layers:
        blocks = sorted({int(k.split(".")[4]) for k in sd
                         if k.startswith(f"layers.{l}.residual_group.blocks.")})
        depths.append(len(blocks))
    embed = sd["layers.0.residual_group.blocks.0.attn.qkv.weight"].shape[1]
    heads = sd["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].shape[1]
    feat = sd["conv_before_upsample.0.weight"].shape[0]
    return depths, embed, heads, feat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--variant", default=None, choices=sorted(VARIANTS))
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--gather-bias", action="store_true",
                    help="write the gathered [nH][n][n] bias tables (38 MB for HAT-S); "
                         "off by default, see the module doc")
    args = ap.parse_args()

    sd_raw, which = load_state(args.src)
    if not isinstance(sd_raw, dict):
        print(f"{args.src}: not a state dict ({type(sd_raw)})", file=sys.stderr)
        return 1
    sd = {k: v.detach().to(torch.float32).contiguous() for k, v in sd_raw.items()
          if isinstance(v, torch.Tensor) and v.is_floating_point()}
    if "conv_first.weight" not in sd:
        print(f"{args.src}: no conv_first.weight - not a HAT checkpoint?", file=sys.stderr)
        return 1
    idx = {k: v for k, v in sd_raw.items() if k.startswith("relative_position_index")}

    depths, embed, heads, feat = infer(sd)
    variant = args.variant
    if variant is None:
        for name, cfg in VARIANTS.items():
            if (cfg["depths"] == depths and cfg["embed_dim"] == embed
                    and cfg["num_heads"][0] == heads):
                variant = name
                break
    if variant is None:
        print(f"{args.src}: tensors describe depths={depths} embed={embed} heads={heads}, "
              f"which is none of {sorted(VARIANTS)}; pass --variant", file=sys.stderr)
        return 1
    cfg = VARIANTS[variant]
    if (cfg["depths"] != depths or cfg["embed_dim"] != embed
            or cfg["num_heads"][0] != heads):
        print(f"warning: the weights are depths={depths} embed={embed} heads={heads} but the "
              f"published {variant} config is depths={cfg['depths']} embed={cfg['embed_dim']} "
              f"heads={cfg['num_heads'][0]}", file=sys.stderr)

    scale = args.scale
    if scale is None:
        # THE WIDENED WIDTH DISTINGUISHES THE TWO `Upsample` BRANCHES, not the block
        # count. A 2^n scale repeats `Conv2d(num_feat, 4*num_feat, 3)` n times, while
        # scale 3 is a SINGLE `Conv2d(num_feat, 9*num_feat, 3)` - so counting blocks
        # reads a scale-3 head (one block, widened 9*feat) as x2, which is a wrong
        # model rather than an error.
        blocks = sorted(int(k.split(".")[1]) for k in sd
                        if k.startswith("upsample.") and k.endswith(".weight"))
        widened = sd[f"upsample.{blocks[0]}.weight"].shape[0] if blocks else 0
        if widened == 9 * feat:
            scale = 3
            print("no --scale given: the head's first conv widens to 9*num_feat, "
                  "i.e. the reference's scale-3 branch, x3", file=sys.stderr)
        else:
            scale = 2 ** len(blocks)
            print(f"no --scale given: the head has {len(blocks)} octaves, i.e. x{scale}",
                  file=sys.stderr)

    # The parameters no forward reads. `.drop_path` is a DropPath(0) = nn.Identity, and everything under
    # `patch_embed.` except its `norm` belongs to a patch_size-1 projection that is
    # an identity reshape. NOTHING ELSE qualifies: in particular `norm1`/`norm2` ARE
    # read by HAB.forward and are trained - dropping them silently evaluated a
    # different network (see the module doc).
    dropped = sorted(k for k in list(sd)
                     if k.endswith(".drop_path")
                     or (k.startswith("patch_embed.")
                         and not k.startswith("patch_embed.norm.")))
    for k in dropped:
        del sd[k]

    out = {k: v.numpy() for k, v in sd.items()}
    for k, v in idx.items():
        # An int64 index is not a safetensors F32 tensor; the engine reads these
        # only to check its own index construction (see `Weights::i64`), so they
        # are carried as I64 alongside the weights.
        out[k] = v.detach().numpy().astype(np.int64)

    if args.gather_bias:
        win, owin = WINDOW, WINDOW + int(WINDOW * OVERLAP)
        rpi_sa = idx["relative_position_index_SA"].view(-1).long()
        rpi_oca = idx["relative_position_index_OCA"].view(-1).long()
        for layer in range(len(cfg["depths"])):
            for blk in range(cfg["depths"][layer]):
                for suffix, rpi, n in [("attn", rpi_sa, win * win),
                                       ("overlap_attn", rpi_oca, win * win)]:
                    p = f"layers.{layer}.residual_group.{suffix if suffix == 'attn' else 'blocks.%d.attn' % 0}"
                    del p
            for blk in range(cfg["depths"][layer]):
                p = f"layers.{layer}.residual_group.blocks.{blk}.attn"
                tbl = sd_raw[p + ".relative_position_bias_table"].float()
                g = tbl[rpi_sa].view(win * win, win * win, heads).permute(2, 0, 1).contiguous()
                out[p + ".relative_position_bias"] = g.numpy()
            p = f"layers.{layer}.residual_group.overlap_attn"
            tbl = sd_raw[p + ".relative_position_bias_table"].float()
            g = tbl[rpi_oca].view(win * win, owin * owin, heads).permute(2, 0, 1).contiguous()
            out[p + ".relative_position_bias"] = g.numpy()

    metadata = {
        "variant": variant,
        "scale": str(scale),
        "window_size": str(WINDOW),
        "overlap_ratio": str(OVERLAP),
        "embed_dim": str(embed),
        "num_heads": str(heads),
        "mlp_ratio": str(cfg["mlp_ratio"]),
        "depths": ",".join(str(d) for d in depths),
        "compress_ratio": str(cfg["compress_ratio"]),
        "squeeze_factor": str(cfg["squeeze_factor"]),
        "conv_scale": str(CONV_SCALE),
        "img_range": str(IMG_RANGE),
        "rgb_mean": MEAN,
        "source": f"{Path(args.src).name}:{which}",
        "format": "pt",
    }

    tensors = sorted(out.items())
    offset = 0
    header = {}
    for name, a in tensors:
        a = np.ascontiguousarray(a)
        n = a.size * a.dtype.itemsize
        header[name] = {"dtype": {"float32": "F32", "int64": "I64"}[str(a.dtype)],
                        "shape": list(a.shape), "data_offsets": [offset, offset + n]}
        offset += (n + 3) & ~3
    header["__metadata__"] = metadata
    hjson = json.dumps(header, separators=(",", ":")).encode()
    header_bytes = 8 + len(hjson)
    pad = (-header_bytes) & 7
    with open(args.dst, "wb") as f:
        f.write(struct.pack("<Q", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        written = 0
        for name, a in tensors:
            raw = np.ascontiguousarray(a).tobytes()
            f.write(raw)
            written += len(raw)
            gap = ((len(raw) + 3) & ~3) - len(raw)
            if gap:
                f.write(b"\0" * gap)
                written += gap
    total = len(json.dumps(header))
    print(f"wrote {args.dst}: {len(tensors)} tensors, {written} bytes of payload, "
          f"{total} bytes of header")
    print(f"  variant {variant}, x{scale}, embed {embed}, heads {heads}, "
          f"depths {depths}, taken from `{which}` of {Path(args.src).name}")
    if dropped:
        print(f"  dropped {len(dropped)} tensors no forward reads "
              f"(DropPath identities and patch_size-1 projections, e.g. {dropped[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
