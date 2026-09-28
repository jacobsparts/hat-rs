//! The memory refusal guard: refuse a pass BEFORE its buffers are allocated.
//!
//! WHY A GUARD AND NOT AN ERROR PATH. Both backends allocate their whole working
//! set in one go when the plan changes. On the CPU a failed `Vec` allocation does
//! not return an error - it aborts, because the release profile sets
//! `panic = "abort"` - so there is no failure to report from inside the allocation
//! and no way to recover from it. And a pass that is admitted and does not fit does
//! not fail cleanly either: it evicts the machine's page cache and starts swapping,
//! which is slower than a refusal by orders of magnitude and takes every other
//! process on the machine down with it. The one place the answer is still cheap is
//! BEFORE the allocation, which is what this module is for.
//!
//! `MemAvailable` AND NOT `MemFree`. Free memory is the part that is already
//! unused, which excludes the kernel's reclaimable page cache - so guarding on it
//! refuses passes that would have run comfortably. `MemAvailable` is the kernel's
//! own estimate of what can be had WITHOUT SWAPPING, which is exactly the question,
//! and the property that makes the refusal thrash-proof: a run that is admitted is
//! one the kernel believes it can satisfy from RAM.
//!
//! SWAP IS DELIBERATELY NOT ADDED IN. `MemAvailable` already assumes no swapping, so
//! counting `SwapFree` towards the budget would admit exactly the runs this guard
//! exists to prevent - a pass that "fits" only by paging 6 GB to a disk. A machine
//! with swap is not a machine with more memory, only one that fails more slowly.
//!
//! THE NUMBERS ARE THE ENGINE'S OWN DERIVATION, NOT A FITTED CONSTANT. Both
//! backends already derive every buffer a forward allocates (`Cpu::footprint_floats`
//! and `gpu::footprint_floats`), and on the CPU a test holds that derivation to the
//! buffers actually allocated. Measured against peak RSS over three checkpoints and
//! four sizes, `footprint + the checkpoint's size` was an over-estimate at every one
//! of the twelve points, so the guard refuses on a quantity that is already
//! conservative before `SLACK` is applied.
//!
//! WHAT IT DOES NOT COVER. This is a model, not a measurement, and it is a
//! prediction of a machine's state rather than a reservation: two processes can pass
//! the same check and then not both fit. The virtue it does have is that the failure
//! is at the front, with a number attached, instead of in the middle of an 80-second
//! forward.

/// Headroom over the derived requirement, as a multiplier. Small, because the
/// derivation is already an over-estimate at every point measured; this covers the
/// allocator's own slack (a `Vec` that doubles) rather than the model's error.
pub const SLACK: f64 = 1.05;

/// The part of a forward that is not modelled: the binary and its pages, the rayon
/// pool's stacks, the CUDA context, and the loader. The measured CUDA context alone
/// is 104 MiB (idle 514 MiB -> 618 MiB across a selftest that allocates nothing
/// large), and 32x32 - the smallest pass the engine runs - has a 37 MiB gap between
/// its derived footprint and its peak RSS.
pub const BASE: usize = 128 << 20;

/// Bytes available for a large allocation, from `MemAvailable`.
///
/// `None` means the question could not be answered - an unusual platform, a
/// restricted `/proc` - and every caller treats that as "do not guard" rather than
/// "refuse". A guard that fails closed on a machine it cannot read would be worse
/// than no guard: it would break the engine where it previously worked, and it would
/// be wrong for exactly the reason it cannot check.
pub fn host_available() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        // `strip_prefix` rather than a split: `MemTotal:` also CONTAINS `Mem`, and
        // `/proc/meminfo`'s first line is `MemTotal:` - so a `starts_with("Mem")`
        // style test or a contains-scan reads the wrong line on every machine.
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
        return Some(kb * 1024);
    }
    None
}

/// The soft `RLIMIT_AS`, when it is set, from `/proc/self/limits`.
///
/// `MemAvailable` knows nothing about this limit, so a process running under
/// `ulimit -v` would be admitted by the check above and then fail on an allocation
/// it was never allowed to make. `RLIMIT_AS` is a hard failure rather than a
/// slowdown, so it is a second ceiling and the lower of the two is the real one.
/// `None` means unlimited, which is the normal case here.
pub fn address_space_limit() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/self/limits").ok()?;
    let line = text.lines().find(|l| l.starts_with("Max address space"))?;
    let soft = line.get("Max address space".len()..)?.split_whitespace().next()?;
    if soft == "unlimited" {
        return None;
    }
    soft.parse::<usize>().ok()
}

/// `1.4 GiB`, `812 MiB`. Binary units, because that is what the numbers compared
/// against (`MemAvailable`, a derived float count) are actually in.
pub fn fmt_bytes(n: usize) -> String {
    const MIB: f64 = 1048576.0;
    let mib = n as f64 / MIB;
    if mib >= 1024.0 {
        format!("{:.1} GiB", mib / 1024.0)
    } else {
        format!("{mib:.0} MiB")
    }
}

/// What a pass needs and what is available, so the refusal can quote both.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// The activations: the buffers a forward allocates for this geometry.
    pub activations: usize,
    /// The checkpoint's resident bytes. On the device this is the uploaded tensor
    /// total; on the host it is the mmap'd file, which the page cache holds.
    pub weights: usize,
    /// The unmodelled part, [`BASE`].
    pub base: usize,
    /// `activations * SLACK + weights + base` - the number compared.
    pub need: usize,
    /// What the machine said it had, or `None` if it would not say.
    pub avail: Option<usize>,
}

fn plan_of(activations_floats: u64, weights: u64, avail: Option<usize>) -> Plan {
    let activations = (activations_floats * 4) as usize;
    let weights = weights as usize;
    let need = (activations as f64 * SLACK) as usize + weights + BASE;
    // The address-space limit is a ceiling the kernel will enforce regardless of
    // free memory, so it takes the place of `avail` when it is the lower of the two.
    let avail = match (avail, address_space_limit()) {
        (Some(a), Some(l)) => Some(a.min(l)),
        (a, None) => a,
        (None, l) => l,
    };
    Plan { activations, weights, base: BASE, need, avail }
}

/// The CPU plan for a `h`x`w` input, from the CPU backend's own derivation.
pub fn cpu_plan(wt: &crate::weights::Weights, h: usize, w: usize) -> Plan {
    let plan = crate::plan::Plan::new(h, w, wt.window, wt.embed);
    let floats = crate::cpu::Cpu::footprint_floats(wt, &plan);
    plan_of(floats, wt.bytes, host_available())
}

/// Refuse a CPU pass that does not fit, before its buffers exist.
///
/// CALLED FROM INSIDE THE BACKEND rather than from the CLI, for two reasons: the
/// allocation is what is being guarded, so the guard belongs at the allocation; and
/// the tiled path calls the backend once per TILE, so a check here is automatically
/// a check of the tile's footprint - which is the number that matters when a user
/// has asked for `--mem`. A guard in `main.rs` would be a guard of the whole image
/// and would be wrong for exactly the case where memory is tight.
pub fn check_cpu(wt: &crate::weights::Weights, h: usize, w: usize) -> Result<Plan, String> {
    let p = cpu_plan(wt, h, w);
    if let Some(avail) = p.avail {
        if p.need > avail {
            return Err(format!(
                "not enough memory for a {w}x{h} pass on the CPU\n\
                 hat: it needs about {} ({} of buffers + {} of weights + {} of runtime, \
                 plus {}% of allocator slack); {} is available\n\
                 hat: a smaller image, `--mem` to run it in tiles, or a narrower \
                 checkpoint is what fits - `--mem` trades time for memory and needs \
                 no less than about {} here",
                fmt_bytes(p.need), fmt_bytes(p.activations), fmt_bytes(p.weights),
                fmt_bytes(p.base), ((SLACK - 1.0) * 100.0).round() as usize,
                fmt_bytes(avail),
                fmt_bytes(p.base + p.weights + (p.activations as f64 * SLACK) as usize / 4),
            ));
        }
    }
    Ok(p)
}

/// The device plan for a `h`x`w` input: the checkpoint's uploaded bytes plus the
/// device buffer set, against the driver's free VRAM.
#[cfg(feature = "cuda")]
pub fn gpu_plan(wt: &crate::weights::Weights, h: usize, w: usize) -> Result<Plan, String> {
    let plan = crate::plan::Plan::new(h, w, wt.window, wt.embed);
    let floats = crate::gpu::footprint_floats(wt, &plan);
    let avail = lightgpu::vm::free_vram().ok();
    Ok(plan_of(floats, wt.bytes, avail))
}

/// Refuse a GPU pass that does not fit in free VRAM, with the number the driver
/// gave. Without this the failure is `cuMemAlloc failed: CUDA_ERROR_OUT_OF_MEMORY`
/// from somewhere in the middle of a forward, naming neither the size that was
/// refused nor the memory that was there.
#[cfg(feature = "cuda")]
pub fn check_gpu(wt: &crate::weights::Weights, h: usize, w: usize) -> Result<Plan, String> {
    let p = gpu_plan(wt, h, w)?;
    if let Some(avail) = p.avail {
        if p.need > avail {
            return Err(format!(
                "not enough video memory for a {w}x{h} pass on the GPU\n\
                 hat: it needs about {} ({} of device buffers + {} of weights + {} of \
                 runtime, plus {}% of slack); {} is free on the device\n\
                 hat: `--mem` runs it in tiles on the device (each tile allocates its \
                 own buffers, so the tile's size is the number that must fit); \
                 a smaller image does too",
                fmt_bytes(p.need), fmt_bytes(p.activations), fmt_bytes(p.weights),
                fmt_bytes(p.base), ((SLACK - 1.0) * 100.0).round() as usize,
                fmt_bytes(avail),
            ));
        }
    }
    Ok(p)
}

/// The checkpoint alone, before it is uploaded.
///
/// `Gpu::new` uploads every f32 tensor in the file and knows nothing about the
/// input size, so this is the one memory question that can be asked at that point -
/// and it is worth asking, because a card that cannot hold the WEIGHTS fails during
/// the upload, before a single forward is attempted, and the driver's
/// `CUDA_ERROR_OUT_OF_MEMORY` does not say which of the two was too big.
#[cfg(feature = "cuda")]
pub fn check_gpu_weights(wt: &crate::weights::Weights) -> Result<(), String> {
    let need = wt.bytes as usize + BASE;
    let Some(avail) = lightgpu::vm::free_vram().ok() else {
        return Ok(());
    };
    if need > avail {
        return Err(format!(
            "not enough video memory for the checkpoint alone\n\
             hat: {} uploads as {} of tensors, and {} is free on the device\n\
             hat: a smaller checkpoint is what fits",
            wt.variant, fmt_bytes(wt.bytes as usize), fmt_bytes(avail),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Option<String> {
        let p = "../models/hat-s-x4.safetensors".to_string();
        std::path::Path::new(&p).exists().then_some(p)
    }

    #[test]
    fn host_available_reads_the_line_it_means() {
        // `/proc/meminfo` opens with `MemTotal:`, which starts with the same three
        // letters as `MemAvailable:` - so this asserts the parse found AVAILABLE and
        // not the first line that looks close. `MemAvailable` is the smaller of the
        // two by a large margin on a machine with any page cache at all.
        let a = host_available().expect("/proc/meminfo is readable on Linux");
        assert!(a > 0);
        let text = std::fs::read_to_string("/proc/meminfo").unwrap();
        let total: usize = text
            .lines()
            .find_map(|l| l.strip_prefix("MemTotal:")?.split_whitespace().next()?.parse().ok())
            .unwrap();
        assert!(a < total * 1024, "available {a} must be below total {}", total * 1024);
    }

    #[test]
    fn fmt_bytes_is_binary() {
        assert_eq!(fmt_bytes(1024 * 1024), "1 MiB");
        assert_eq!(fmt_bytes(1536 * 1024 * 1024), "1.5 GiB");
    }

    #[test]
    fn a_pass_beyond_the_machine_is_refused_before_it_allocates() {
        let Some(model) = model() else {
            eprintln!("SKIP: no model (set HAT_MODEL to a converted checkpoint)");
            return;
        };
        let wt = crate::weights::Weights::load(&model).expect("load the checkpoint");
        // A 32x32 pass is a few tens of MiB and must pass on any machine that can
        // run this test at all.
        let ok = check_cpu(&wt, 32, 32).expect("a 32x32 pass fits anywhere this test runs");
        assert!(ok.need > ok.activations, "the plan counts weights and runtime too");
        // 8192x8192 is 256 times the pixels and cannot fit: it must be refused, and
        // the refusal must quote the numbers rather than say "failed".
        let big = check_cpu(&wt, 8192, 8192).expect_err("8192x8192 cannot fit");
        assert!(big.contains("not enough memory"), "{big}");
        assert!(big.contains("--mem"), "a refusal must name a remedy: {big}");
        // And the guard is only a refusal when the machine answered at all.
        if host_available().is_none() {
            assert!(check_cpu(&wt, 8192, 8192).is_ok(), "no reading, no refusal");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_device_plan_is_the_device_buffer_list() {
        let Some(model) = model() else {
            eprintln!("SKIP: no model (set HAT_MODEL to a converted checkpoint)");
            return;
        };
        let wt = crate::weights::Weights::load(&model).expect("load the checkpoint");
        let plan = crate::plan::Plan::new(48, 48, wt.window, wt.embed);
        // The derivation and the constructor are the same list, so a GPU plan is
        // never a second opinion about the buffer set.
        let derived = crate::gpu::footprint_floats(&wt, &plan);
        assert!(derived > 0);
        assert!(derived > (10 * plan.tokens() * wt.embed) as u64,
                "the buffer set is more than ten c-channel planes");
    }
}
