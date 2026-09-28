# hat-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

HAT (Hybrid Attention Transformer) image super-resolution in one self-contained
binary, at 2x, 3x and 4x. No Python, PyTorch, ONNX Runtime, or CUDA toolkit
needed.

```
hat -m hat-s-x4.safetensors -i small.png -o large.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. `--device cpu|gpu` picks between them, and the CPU is the
  default because a GPU that has to be fed a pass it cannot hold is slower than
  the CPU that can - a 64x64 image takes 1.4 s on the CPU and 0.26 s on the GPU,
  512x512 takes 82 s and 8.6 s.
* 2.25 MiB binary (1.19 MiB without the CUDA feature), statically linked except
  `libc`, `libm` and `libgcc_s`. `libcuda.so.1` is loaded at run time, so no
  driver is needed on disk, and `--no-default-features` builds a binary with no
  CUDA in it at all - which is also what lets `cargo build` work on a machine
  with no toolkit installed.
* Three published HAT checkpoints, converted from the official `.pth` files, plus
  a 3x one that no release provides (see Models below).

Both backends reproduce the upstream PyTorch implementation: the CPU to within
about 5e-6 and the GPU to within about 1e-5 of the reference's own fp32 output,
against a tolerance of 1e-4.

## Download

Prebuilt binary and the converted checkpoints are attached to the
[release](https://github.com/jacobsparts/hat-rs/releases).

| asset | what it is |
|---|---|
| `hat-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+). The CUDA kernels are compiled in for compute capability 6.1, 7.5 and 8.0; with no driver, `--device cpu` is the path |
| `hat-s-x4.safetensors` | 4x, the small model - **speed** |
| `hat-x4.safetensors` | 4x, the base model |
| `hat-l-x4.safetensors` | 4x, the large model - **quality** |

```sh
chmod +x hat-linux-x86_64
./hat-linux-x86_64 -m hat-s-x4.safetensors -i small.png -o large.png
```

## Models

Converted from the official `.pth` files (see `tools/convert.py`); `-m` is the
whole choice. HAT-S trades a little quality for a lot of speed, HAT-L is the
slowest and the best, and the difference between them is a few tenths of a dB on
a benchmark rather than anything visible on a photograph - which is why `hat-s`
is the one to reach for.

| checkpoint | what it is | embed dim | depth | tensors |
|---|---|---|---|---|
| `hat-s-x4` | 4x, small - **speed** | 144 | 6 per stage, 6 stages | 864 |
| `hat-x4` | 4x, base | 180 | 6 per stage, 6 stages | 864 |
| `hat-l-x4` | 4x, large - **quality** | 180 | 12 per stage, 12 stages | 1710 |

All three use a 16x16 attention window, an overlap ratio of 0.5 and an MLP ratio
of 2, and are converted from the checkpoints' `params_ema` - the released
`params` and `params_ema` differ, and every released configuration is evaluated
with `params_ema`.

The upstream authors' own PSNR figures are not repeated here, because this
repository does not re-measure them. What is measured here is that each backend
reproduces the reference's output, which is a different claim.

**Scale 3 has no published checkpoint.** XPixelGroup released only 4x, so
`hat-s-x3.safetensors` is built from the reference's own code with seeded random
weights - useful for exercising the scale-3 head, not for restoring an image.
Scale 3 is not a count of a 4x block: a power-of-two scale repeats
`Conv2d(64, 4*64, 3)` + `PixelShuffle(2)` once per octave, while scale 3 is a
single `Conv2d(64, 9*64, 3)` + `PixelShuffle(3)`. `tests/parity.rs` records the
exact commands that reproduce it.

## Usage

```sh
hat -m hat-s-x4.safetensors -i small.png -o large.png
hat -m hat-s-x4.safetensors -i small.png -o large.png --device gpu
hat -m hat-s-x4.safetensors --verify tests/data/hats_x4_48x48.bin
hat -m hat-s-x4.safetensors -i huge.png -o large.png --mem 2000
hat --cuda-selftest
```

```
hat - HAT super-resolution (S/M/L, x2/x3/x4)

USAGE:
    hat -m <model.safetensors> -i <in.png> -o <out.png> [--device cpu]
    hat -m <model.safetensors> --verify <fixture.bin>
    hat -m <model.safetensors> --list-weights
    hat --cuda-selftest

OPTIONS:
    -m, --model <path>    a checkpoint converted by tools/convert.py
    -i, --input <path>    the image to upscale (PNG)
    -o, --output <path>   where to write the result (PNG)
        --verify <path>   compare a backend against a golden fixture and report the
                          worst and mean absolute difference; exits 1 if it is over
                          the tolerance, so it can be used as a test
        --device <name>   cpu (default) or gpu
        --list-weights    print every tensor the checkpoint holds, with its shape
        --cuda-selftest   check every project CUDA kernel against a host implementation
                          of the same operator, on seeded inputs (no -m needed); a
                          kernel whose arguments are mis-ordered still produces a
                          plausible image, and this is what catches it
        --tile <px>       run the image in <px>-wide tiles (rounded up to a window
                          multiple) instead of one pass, for images that do not fit
                          in memory; the tile is approximate near its edges, since a
                          window attention reaches across the whole window and each
                          3x3 conv widens the field by a pixel
        --mem <MiB>       a working-set budget; chooses the largest window-multiple
                          tile that fits it, and caps `--tile` if both are given
    -h, --help            this text
```

`--verify` and `--cuda-selftest` are the correctness story. `--verify` compares a
backend's whole image against the published network's own output, which catches a
wrong graph. `--cuda-selftest` runs every project kernel against a Rust
implementation of the same operator on seeded inputs, which is what catches a
kernel whose shape arguments are mis-ordered - a mistake that produces a
plausible image rather than a fault, and one the image comparison cannot see.

## Large images

Both backends allocate their whole working set in one go, so memory grows with
the input, and the engine counts every buffer a pass will use before it allocates
any of them. The requirement is therefore a number:

| input | pass buffers, hat-s | pass buffers, hat / hat-l |
| --- | --- | --- |
| 128x128 | 424 MiB | 488 MiB |
| 256x256 | 1.7 GiB | 1.9 GiB |
| 384x384 | 3.8 GiB | 4.4 GiB |
| 512x512 | 6.8 GiB | 7.8 GiB |
| 1024x1024 | 27.1 GiB | 31.2 GiB |
| 2048x2048 | 108 GiB | 125 GiB |

Those are the pass's buffers, and they depend on the geometry rather than on the
number of blocks: `hat` and `hat-l` are identical here, because what is allocated
is a set of per-stage planes and HAT-L's extra depth does not widen them. Add the
checkpoint (40 MB for hat-s, 158 MB for hat-l), 128 MiB of runtime, and 5% of
slack for the allocator to get the figure the engine compares against the machine.

A pass that does not fit is refused before it allocates anything, with both
numbers and a way out:

```
hat: not enough memory for a 4096x4096 pass on the CPU
hat: it needs about 444.1 GiB (422.8 GiB of buffers + 38 MiB of weights + 128 MiB of runtime, plus 5% of allocator slack); 31.8 GiB is available
hat: a smaller image, `--mem` to run it in tiles, or a narrower checkpoint is what fits - `--mem` trades time for memory and needs no less than about 111.1 GiB here
```

```
hat: not enough video memory for a 512x512 pass on the GPU
hat: it needs about 7.9 GiB (7.2 GiB of device buffers + 158 MiB of weights + 128 MiB of runtime, plus 5% of slack); 7.2 GiB is free on the device
```

The guard reads `MemAvailable`, not `MemFree`, and deliberately does not count
swap: a pass that fits only by paging is a pass that will thrash the machine for
an hour instead of being refused in a millisecond.

`--mem` is the way through when a pass does not fit. It runs the image in
window-aligned tiles, each of which allocates its own buffers, so the tile's
footprint is what has to fit - and because the budget is checked against the
sub-image including its halo, the flag is a bound rather than a hint:

```sh
hat -m hat-s-x4.safetensors -i huge.png -o large.png --mem 2000
hat -m hat-s-x4.safetensors -i huge.png -o large.png --mem 2000 --device gpu
```

A tile is approximate near its edges - a window attention reaches across its
whole window and each 3x3 convolution widens the field by a pixel - but a tile
that covers the image is not a tile, and that case reproduces the whole-image
result bit for bit.

## Building

```sh
cargo build --release                         # both backends; needs nvcc for CUDA
cargo build --release --no-default-features   # pure Rust, no CUDA
```

`nvcc` compiles `cuda/hat.cu` and the toolkit subset it calls into two fatbins;
set `NVCC=/path/to/nvcc` if it is not on `PATH`. The build checks each fatbin's
kernel list against its source in both directions, so a kernel that is written
but not exported - or exported but no longer written - fails the build rather
than the run.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE). This is an independent reimplementation of the HAT
architecture, which is by [XPixelGroup](https://github.com/XPixelGroup/HAT) and
Apache-2.0 licensed. The converted `.safetensors` checkpoints are format
conversions of the official `.pth` files, redistributed under the same terms. The
original `.pth` files are not redistributed here.
