//! The `hat` binary: one image in, one image out.
//!
//! The library does the work; this is the argument parsing, the file I/O and the
//! reporting. The flag set is the one the family uses, so a command line written for
//! the sibling engines works here.
use std::process::ExitCode;

use hat::backend::Backend;
use hat::cpu::Cpu;
use hat::fixture::Fixture;
use hat::image;
use hat::weights::Weights;

struct Args {
    model: String,
    input: Option<String>,
    output: Option<String>,
    verify: Option<String>,
    device: String,
    list_weights: bool,
    /// `--cuda-selftest`: check every project CUDA kernel against its host twin.
    /// Requires no model - it runs on synthetic inputs.
    selftest: bool,
    /// `--tile N`: run the image in N-pixel cores. `None` means one pass.
    tile: Option<usize>,
    /// `--mem N`: a working-set budget in MiB; picks the tile automatically.
    mem: Option<f64>,
}

fn usage() -> &'static str {
    "hat - HAT super-resolution (S/M/L, x2/x3/x4)

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
    -h, --help            this text"
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        input: None,
        output: None,
        verify: None,
        device: "cpu".to_string(),
        list_weights: false,
        selftest: false,
        tile: None,
        mem: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-m" | "--model" => a.model = next()?,
            "-i" | "--input" => a.input = Some(next()?),
            "-o" | "--output" => a.output = Some(next()?),
            "--verify" => a.verify = Some(next()?),
            "--device" => a.device = next()?,
            "--list-weights" => a.list_weights = true,
            "--cuda-selftest" => a.selftest = true,
            "--tile" => {
                a.tile = Some(next()?.parse().map_err(|e| format!("--tile: {e}"))?);
            }
            "--mem" => {
                a.mem = Some(next()?.parse().map_err(|e| format!("--mem: {e}"))?);
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
        }
    }
    // `--cuda-selftest` runs on synthetic inputs, so it needs no checkpoint.
    if a.model.is_empty() && !a.selftest {
        return Err(format!("no model: -m is required\n\n{}", usage()));
    }
    Ok(a)
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hat: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    if args.selftest {
        return selftest();
    }
    let wt = Weights::load(&args.model)?;
    println!("hat {} - {} (x{}), {} bytes",
             hat::VERSION, wt.variant, wt.scale, wt.bytes);
    if args.list_weights {
        for n in &wt.names {
            let s = wt.shape(n);
            println!("  {n}  {s:?}");
        }
        return Ok(());
    }

    if let Some(path) = &args.verify {
        return verify(&wt, &args.device, path);
    }

    let (input, output) = match (&args.input, &args.output) {
        (Some(i), Some(o)) => (i, o),
        _ => return Err("nothing to do: give -i and -o, or --verify".to_string()),
    };
    let img = image::load_rgb(input)?;
    let (w, h) = (img.w, img.h);
    println!("  input  {w}x{h} -> {}x{}", w * wt.scale, h * wt.scale);
    let t0 = std::time::Instant::now();
    let planes = if args.tile.is_some() || args.mem.is_some() {
        let margin = default_margin(&wt);
        // `--mem` picks the tile from the backend's own accounting; an explicit
        // `--tile` is honoured but still capped by the budget when both are given,
        // because the budget is the promise and the tile is the preference.
        let tile = match (args.tile, args.mem) {
            (Some(t), None) => t,
            (t, Some(mib)) => {
                let budget = (mib * 1024.0 * 1024.0) as u64;
                let plan = hat::plan::Plan::new(h, w, wt.window, wt.embed);
                let per_pixel = Cpu::footprint_floats(&wt, &plan) / (plan.tokens().max(1) as u64);
                let auto = hat::tile::auto_tile(h, w, wt.window, budget, per_pixel, 0, margin)
                    .ok_or_else(|| format!("--mem {mib} MiB is too small even for one window (the index maps alone do not shrink)"))?;
                println!("  --mem {mib} MiB -> tile {auto}px (margin {margin})");
                match t { Some(t) => t.min(auto), None => auto }
            }
            (None, None) => unreachable!("guarded by the outer if"),
        };
        let (out, info) = tile_forward(&wt, &args.device, h, w, &img.data, tile, margin)?;
        println!("  tiled {} into {} tiles ({}px cores, {}px margin, largest {}x{})",
                 format!("{}x{}", w, h), info.tiles, info.tile, info.margin,
                 info.largest.0, info.largest.1);
        out
    } else {
        dispatch(&wt, &args.device, h, w, &img.data)?
    };
    let elapsed = t0.elapsed();
    println!("  {} forward in {:.1} ms ({:.2} MP/s)", args.device,
             elapsed.as_secs_f64() * 1e3,
             (w * h) as f64 / 1e6 / elapsed.as_secs_f64());
    let out = image::Image { w: w * wt.scale, h: h * wt.scale, data: planes };
    image::save_rgb(output, out.w, out.h, &out.to_rgb8())?;
    println!("  wrote {output}");
    Ok(())
}

/// The halo a tile is given, in input pixels, rounded up to a window by the tile
/// loop. It is DERIVED from the network rather than tuned: every 3x3 convolution
/// widens the receptive field by one pixel, and there is one per HAB's CAB (two), one
/// per HAB's MLP (none - it is 1x1), one per OCAB's two CAB convolutions, one per
/// RHAG and one per stage's downsample-free tail. Counting them exactly would be
/// fragile; counting generously costs only time. Two windows covers the largest
/// non-local reach (a shifted window attention sees one window either side) plus
/// every conv's single pixel, which is what the margin is for.
fn default_margin(wt: &Weights) -> usize {
    2 * wt.window
}

/// The mean absolute difference a fixture is expected to agree within. The two
/// backends accumulate in a different order from the reference, so this is a
/// floating-point budget, not a claim of exactness.
const VERIFY_TOL: f32 = 1e-4;

fn verify(wt: &Weights, device: &str, path: &str) -> Result<(), String> {
    let fx = Fixture::load(path)?;
    if fx.scale != wt.scale {
        return Err(format!("{} is a x{} fixture but the model is x{}", path, fx.scale, wt.scale));
    }
    let planes = dispatch(wt, device, fx.h, fx.w, &fx.input)?;
    let (worst, at, mean) = fx.compare(&planes);
    let (y, x, c) = fx.locate(at);
    println!("  verify worst |diff| {worst:.3e} at (y {y}, x {x}, ch {c}), mean |diff| {mean:.3e}");
    if worst > VERIFY_TOL {
        return Err(format!("worst |diff| {worst:.3e} exceeds {VERIFY_TOL:.0e}"));
    }
    println!("  ok");
    Ok(())
}

fn dispatch(wt: &Weights, device: &str, h: usize, w: usize, input: &[f32])
    -> Result<Vec<f32>, String> {
    match device {
        "cpu" => Cpu::new(wt)?.forward(h, w, input),
        "gpu" => gpu_forward(wt, h, w, input),
        other => Err(device_error(other)),
    }
}

/// The TILED branch, which dispatches by device exactly as `dispatch` does. It exists
/// because it did not: the tile loop was called on a hardcoded `Cpu::new(&wt)` while the
/// label printed `args.device`, so `--device gpu --tile 32` ran the CPU for every tile
/// and reported "gpu forward in ...". `tile::forward` is generic over `Backend`, so the
/// only thing that was ever missing was this match - and a tiled run is precisely the
/// case where the device matters most, because a tile's `Acts` is the whole device
/// footprint and tiling is what makes a 512x512 image fit in 8 GiB of VRAM at all.
fn tile_forward(wt: &Weights, device: &str, h: usize, w: usize, input: &[f32],
                tile: usize, margin: usize)
    -> Result<(Vec<f32>, hat::tile::Tiling), String> {
    match device {
        "cpu" => hat::tile::forward(&mut Cpu::new(wt)?, h, w, input, tile, margin,
                                     wt.scale, wt.window),
        "gpu" => tile_forward_gpu(wt, h, w, input, tile, margin),
        other => Err(device_error(other)),
    }
}

/// The one message both dispatchers give for a device this build cannot run, so the
/// tiled and untiled paths cannot drift apart on it.
fn device_error(other: &str) -> String {
    format!("device `{other}`: expected `cpu` or `gpu` (the GPU backend needs the `cuda` \
             feature, which is on by default and needs nvcc at build time)")
}

#[cfg(feature = "cuda")]
fn tile_forward_gpu(wt: &Weights, h: usize, w: usize, input: &[f32], tile: usize, margin: usize)
    -> Result<(Vec<f32>, hat::tile::Tiling), String> {
    hat::tile::forward(&mut hat::gpu::Gpu::new(wt)?, h, w, input, tile, margin,
                       wt.scale, wt.window)
}

#[cfg(not(feature = "cuda"))]
fn tile_forward_gpu(_wt: &Weights, _h: usize, _w: usize, _input: &[f32], _tile: usize,
                    _margin: usize) -> Result<(Vec<f32>, hat::tile::Tiling), String> {
    Err("this build has no `cuda` feature: `--device gpu` needs it (it is on by default; \
         `--no-default-features` builds the pure-Rust engine)"
        .into())
}

/// `--cuda-selftest`, behind the feature: a pure-Rust build has no project kernels
/// to check, and saying so is better than a link error.
#[cfg(feature = "cuda")]
fn selftest() -> Result<(), String> {
    println!("cuda selftest - every project kernel against a host twin");
    hat::selftest::run()
}

#[cfg(not(feature = "cuda"))]
fn selftest() -> Result<(), String> {
    Err("this build has no `cuda` feature, so there are no CUDA kernels to check \
         (it is on by default; `--no-default-features` builds the pure-Rust engine)"
        .into())
}

/// `--device gpu`, behind the feature so a pure-Rust build needs no nvcc. The
/// backend object is built per call rather than kept: `verify` runs a handful of
/// sizes and the weight upload is the cheap part of a forward, while threading a
/// cached backend through `dispatch`'s signature would make every caller hold one.
#[cfg(feature = "cuda")]
fn gpu_forward(wt: &Weights, h: usize, w: usize, input: &[f32]) -> Result<Vec<f32>, String> {
    hat::gpu::Gpu::new(wt)?.forward(h, w, input)
}

#[cfg(not(feature = "cuda"))]
fn gpu_forward(_wt: &Weights, _h: usize, _w: usize, _input: &[f32]) -> Result<Vec<f32>, String> {
    Err("this build has no `cuda` feature: `--device gpu` needs it (it is on by default; \
         `--no-default-features` builds the pure-Rust engine)"
        .into())
}
