#!/usr/bin/env python3
"""The plan oracle: the reference's own window/unfold/mask geometry, as data.

`src/plan.rs` builds HAT's three host-side index structures - the window map (a
`torch.roll` folded into the partition), the OCAB's `nn.Unfold` gather, and the
shifted-window attention mask - and the doc comments there claim specific
conventions for each. Claims in comments are not evidence, so this script records
what PyTorch actually does and `tests/plan.rs` compares the engine against it.

Each structure is derived from the REAL operation, not from a reimplementation:

* the window map comes from `torch.roll` followed by HAT's own `window_partition`,
  applied to a plane whose every value is its own pixel index;
* the OCAB gather comes from `nn.Unfold(kernel_size=owin, stride=win, padding=pad)`
  applied to a plane of `pixel_index + 1`, so a zero marks a padded position and a
  non-zero value names the pixel the gather reads;
* the mask comes from the reference's own `calculate_mask`, stored as the per-window
  LABEL grid it compares (`mask[q][k] = -100 where label[q] != label[k]`). The
  labels are stored rather than the mask because the mask is `nw*win^4` floats - 4 MB
  at 64x64 - while the labels are `hp*wp` u32.

THE INDEXES ARE ONLY MEANINGFUL WHERE `keep == 1`. For a padded position the
reference's gather READS A ZERO from outside the plane; the engine's map stores the
clamped in-plane position there and relies on the `keep` byte to skip the load, so
the two disagree on the index of a masked slot by construction and agree on which
slots are masked. `tests/plan.rs` compares the mask exactly and the indexes only
where it is set.

Usage:
    python3 tools/make_plan_oracle.py --out tests/data/plan_oracle.bin

The container is little-endian: magic `HATPLAN1`, a u32 entry count, then per entry
a u32 name length, the name, a u8 dtype (0 = f32, 1 = u32), a u32 rank, the rank's
u32 dimensions, and the payload.
"""
import argparse
import struct
import numpy as np
import torch

# The geometries the plan is checked at. 16 is the smallest window-multiple, 32 is
# where the shift mask straddles (win + shift = 24 < 32), 48 and 64 are the sizes the
# fixtures and the benchmarks use, and 32x48 and 64x32 are non-square - which matters
# because the plane's two axes are rolled independently.
GEOMETRIES = [(16, 16), (32, 32), (48, 48), (32, 48), (64, 64), (64, 32)]


def window_partition(x, ws):
    """HAT's `window_partition`, verbatim."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)


def window_index(h, w, win, shift):
    """The source pixel of every window token, after `torch.roll(p, -shift)`."""
    x = torch.arange(h * w, dtype=torch.float32).reshape(1, h, w, 1)
    rolled = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
    return window_partition(rolled, win)[..., 0].reshape(-1).numpy().astype(np.uint32)


def unfold_index(h, w, win, owin, pad):
    """`nn.Unfold`'s gather and its padding mask, read off a plane of pixel names."""
    plane = torch.arange(1, h * w + 1, dtype=torch.float32).reshape(1, 1, h, w)
    unf = torch.nn.Unfold(kernel_size=owin, stride=win, padding=pad)
    out = unf(plane)[0]                                  # [owin*owin][nw]
    nw = out.shape[1]
    # The engine's map is window-major with the kernel position innermost.
    vals = out.transpose(0, 1).reshape(-1).numpy()
    keep = (vals != 0).astype(np.uint8)
    idx = np.where(keep == 1, np.maximum(vals.astype(np.int64) - 1, 0), 0).astype(np.uint32)
    assert len(idx) == nw * owin * owin
    return idx, keep


def calculate_labels(h, w, win, shift):
    """The reference's `calculate_mask` label grid, before it is turned into a mask.

    `HAT.calculate_mask` labels nine rectangles of the plane and then compares labels
    inside each window; these are those labels, already partitioned. The comparison
    itself is what `tests/plan.rs` does with them.
    """
    img_mask = torch.zeros((1, h, w, 1))
    h_slices = (slice(0, -win), slice(-win, -shift), slice(-shift, None))
    w_slices = (slice(0, -win), slice(-win, -shift), slice(-shift, None))
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1
    return window_partition(img_mask, win)[..., 0].numpy().astype(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data/plan_oracle.bin")
    ap.add_argument("--window", type=int, default=16)
    args = ap.parse_args()

    win = args.window
    shift = win // 2
    owin = win + win // 2
    pad = (owin - win) // 2

    entries = []
    for (h, w) in GEOMETRIES:
        nw = (h // win) * (w // win)
        entries.append((f"{h}x{w}/window_index_plain", window_index(h, w, win, 0)))
        entries.append((f"{h}x{w}/window_index_shifted", window_index(h, w, win, shift)))
        idx, keep = unfold_index(h, w, win, owin, pad)
        entries.append((f"{h}x{w}/unfold_index", idx))
        entries.append((f"{h}x{w}/unfold_keep", keep))
        entries.append((f"{h}x{w}/mask_labels", calculate_labels(h, w, win, shift)))
        print(f"  {h}x{w}: nw={nw} unfold={len(idx)} labels={h*w} masked={int((keep==0).sum())}")

    with open(args.out, "wb") as f:
        f.write(b"HATPLAN1")
        f.write(struct.pack("<I", len(entries)))
        for name, arr in entries:
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<B", 1))
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", d))
            f.write(arr.astype("<u4").tobytes())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
