//! Inference timing and footprint, per input size.
//!
//! The measurement the README quotes: `Cpu::forward` end to end, which is pad +
//! forward + crop - the whole path a user's image takes, including the padding and
//! the mean adjustment that a large image pays for.
//!
//!     cargo run --release --no-default-features --example bench -- \
//!         -m ../models/hat-s-x4.safetensors --sizes 64,128,256 --iters 5
//!
//! WHAT IS REPORTED, and why each number is there:
//!
//! * `min` and `median` of the per-iteration wall clock. `min` is the least
//!   perturbed sample and is the honest one for comparing kernels; `median` is what
//!   a user mostly sees. Three warm-up runs are the default, because the first
//!   iteration pays for the plan's index maps (which are built once per size, not
//!   per call) and for the page faults of a fresh allocation.
//! * `MP/s` as the throughput the size actually buys.
//! * `peakRSS` from `VmHWM`, which is the host-side high-water mark - the number
//!   that decides whether an image fits, and the one the memory work is judged by.
//!   It is read AFTER the runs of one size, so it is the high-water mark of the
//!   whole process up to that point rather than of a single forward.
//!
//! The input is the deterministic plane `tools/bench_torch.py`'s `raw_input`
//! generates, so a PyTorch run on the same (h, w) sees identical numbers value for
//! value.
//!
//! `--device gpu` runs the same loop through the CUDA backend. THE TWO DEVICES
//! REPORT DIFFERENT FOOTPRINTS AND NEITHER IS COMPARABLE TO THE OTHER: the CPU's is
//! `VmHWM` (host pages the process touched, which includes the engine's own
//! scratch and libc's arenas) and the GPU's is the device allocation the engine
//! asked for, read back from `cuMemGetInfo` as free VRAM before minus free VRAM
//! after - which excludes the driver's own overhead but includes every buffer
//! `Acts` holds. The GPU's number is the one the memory objective is about.
use std::time::Instant;

use hat::backend::Backend;
use hat::cpu::Cpu;
use hat::plan::Plan;
use hat::weights::Weights;

/// Free VRAM in MiB, for the device footprint. `None` in a build without `cuda`
/// (the pure-Rust engine has no device to ask).
#[cfg(feature = "cuda")]
fn free_vram_mib() -> Option<f64> {
    lightgpu::vm::free_vram().ok().map(|b| b as f64 / (1024.0 * 1024.0))
}

#[cfg(not(feature = "cuda"))]
fn free_vram_mib() -> Option<f64> {
    None
}

/// The backend the run asked for, or an error naming how to get one.
fn backend<'a>(
    which: &str,
    wt: &'a Weights,
) -> Result<Box<dyn Backend + 'a>, String> {
    match which {
        "cpu" => Ok(Box::new(Cpu::new(wt)?)),
        #[cfg(feature = "cuda")]
        "gpu" => Ok(Box::new(hat::gpu::Gpu::new(wt)?)),
        #[cfg(not(feature = "cuda"))]
        "gpu" => Err("this build has no `cuda` feature: `--device gpu` needs it".into()),
        other => Err(format!("unknown device `{other}`: cpu or gpu")),
    }
}

/// The golden-ratio fractional sequence, value for value as
/// `tools/bench_torch.py`'s `raw_input`.
fn seq(n: usize) -> Vec<f32> {
    let t = 0.618_033_988_749_894_9f32;
    let mut x = 0.0f32;
    (0..n)
        .map(|_| {
            let v = x;
            x += t;
            if x >= 1.0 {
                x -= 1.0;
            }
            v
        })
        .collect()
}

/// The process's peak resident set, in MiB - `VmHWM`, which is the high-water mark
/// rather than the current occupancy, because a forward's scratch is freed as it
/// goes and the question is what the peak was.
fn peak_rss_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

/// The whole-image footprint the CPU backend allocates for a plan, from the
/// engine's own derivation (`Cpu::footprint_floats`). Used as the `acts MiB` column
/// on the CPU so the number is comparable to `--device gpu`'s measured one.
fn plan_acts_floats(wt: &Weights, plan: &Plan) -> u64 {
    Cpu::footprint_floats(wt, plan)
}

fn parse_sizes(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
}

fn main() -> Result<(), String> {
    let mut model = "../models/hat-s-x4.safetensors".to_string();
    let mut sizes = vec![64usize, 128, 256];
    let mut iters = 3usize;
    let mut warmup = 1usize;
    let mut tile: Option<usize> = None;
    let mut device = "cpu".to_string();

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut next = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "-m" | "--model" => model = next()?,
            "--sizes" => sizes = parse_sizes(&next()?),
            "--iters" => iters = next()?.parse().map_err(|e| format!("--iters: {e}"))?,
            "--warmup" => warmup = next()?.parse().map_err(|e| format!("--warmup: {e}"))?,
            "--tile" => tile = Some(next()?.parse().map_err(|e| format!("--tile: {e}"))?),
            "--device" => device = next()?,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    let wt = Weights::load(&model)?;
    println!("hat {} - {} (x{})", hat::VERSION, wt.variant, wt.scale);
    println!("{:<12} {:>9} {:>9} {:>9} {:>8} {:>10}",
             "input", "min ms", "median ms", "MP/s", "peakRSS", "acts MiB");

    for &side in &sizes {
        let (h, w) = (side, side);
        let img = seq(3 * h * w);
        // BEFORE the backend exists, so the device number includes the `Acts` the
        // first forward allocates - the engine's whole device footprint rather than
        // the part that happens to be transient.
        let vram_before = free_vram_mib();
        let mut be = backend(&device, &wt)?;
        for _ in 0..warmup {
            be.forward(h, w, &img)?;
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let out = be.forward(h, w, &img)?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
            // Consume the result so the compiler cannot elide the pass, and check the
            // shape while doing it - a bench that silently produced nothing would
            // report a very good time.
            if out.len() != 3 * h * wt.scale * w * wt.scale {
                return Err(format!("{}x{} produced {} floats, expected {}",
                                   h, w, out.len(), 3 * h * wt.scale * w * wt.scale));
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = times[0];
        let median = times[times.len() / 2];
        let mps = (h * w) as f64 / 1e6 / (median / 1e3);
        let plan = Plan::new(h, w, wt.window, wt.embed);
        let acts = match &device[..] {
            // The GPU holds `Acts` in device memory, and the free-VRAM difference is
            // the measurement of it rather than a derivation from the buffer list.
            "gpu" => match (vram_before, free_vram_mib()) {
                (Some(a), Some(b)) => a - b,
                _ => 0.0,
            },
            _ => plan_acts_floats(&wt, &plan) as f64 * 4.0 / (1024.0 * 1024.0),
        };
        println!("{:<12} {:>9.1} {:>9.1} {:>9.2} {:>8} {:>10.0}",
                 format!("{h}x{w}"),
                 min, median, mps,
                 peak_rss_mib().map(|m| format!("{m:.0}M")).unwrap_or_else(|| "-".into()),
                 acts);
        let _ = plan;
        let _ = tile;
    }
    Ok(())
}
