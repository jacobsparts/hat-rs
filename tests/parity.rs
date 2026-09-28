//! Parity against the golden fixtures: the engine versus the reference's own output.
//!
//! The fixture is produced by `tools/make_fixture.py` from the PUBLISHED network
//! (upstream `hat_arch.py`, a released config and a released `params_ema`
//! checkpoint), so this test compares the engine against the thing it claims to
//! reimplement rather than against a second copy of its own arithmetic.
//!
//! The model path is taken from `HAT_MODEL` when it is set, and otherwise from the
//! sibling `../models/hat-s-x4.safetensors` the repository is developed against. A
//! missing model SKIPS rather than fails, because the checkpoint is a 40 MB binary
//! that is not in this repository - but a missing FIXTURE fails, since that is a
//! checked-in file and its absence means the test is not running.
use hat::backend::Backend;
use hat::cpu::Cpu;
use hat::fixture::Fixture;
use hat::weights::Weights;

fn model_path() -> Option<String> {
    if let Ok(p) = std::env::var("HAT_MODEL") {
        return Some(p);
    }
    let p = "../models/hat-s-x4.safetensors".to_string();
    std::path::Path::new(&p).exists().then_some(p)
}

/// The SCALE-3 checkpoint, which no released HAT provides - XPixelGroup published
/// only x4 - so it is built from the reference's own code with seeded random weights
/// and is reproducible exactly:
///
/// ```text
/// python3 tools/make_fixture.py --config s --scale 3 --h 32 --w 32 --seed 5 \
///     --save-weights /tmp/hats_x3_random.pth
/// python3 tools/convert.py /tmp/hats_x3_random.pth ../models/hat-s-x3.safetensors \
///     --variant hat-s
/// python3 tools/make_fixture.py --model /tmp/hats_x3_random.pth --config s --scale 3 \
///     --h 48 --w 48 --seed 1 --out tests/data/hats_x3_48x48.bin
/// ```
///
/// WHY IT IS WORTH THIS. Scale 3 is not a different count of the same block: the
/// reference's `Upsample` has two branches, and scale 3 is a SINGLE
/// `Conv2d(num_feat, 9*num_feat, 3)` + `PixelShuffle(3)` where a power of two is n
/// repetitions of `Conv2d(num_feat, 4*num_feat, 3)` + `PixelShuffle(2)`. Nothing
/// else in the suite exercises the 3x shuffle's permutation or a head with one block
/// (and `hat_pixel_shuffle`'s factor is a runtime argument, so a kernel that
/// hardcoded 2 would still write a full, plausible plane).
fn x3_model_path() -> Option<String> {
    if let Ok(p) = std::env::var("HAT_X3_MODEL") {
        return Some(p);
    }
    let p = "../models/hat-s-x3.safetensors".to_string();
    std::path::Path::new(&p).exists().then_some(p)
}

#[test]
fn cpu_matches_the_reference_on_the_48x48_x4_fixture() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no model (set HAT_MODEL to a converted checkpoint)");
        return;
    };
    let fx = Fixture::load("tests/data/hats_x4_48x48.bin").expect("load the golden fixture");
    let wt = Weights::load(&model).expect("load the converted checkpoint");
    assert_eq!(wt.scale, fx.scale, "the checkpoint's scale and the fixture's disagree");
    assert_eq!(wt.window, fx.win, "the checkpoint's window and the fixture's disagree");

    let mut cpu = Cpu::new(&wt).expect("build the CPU backend");
    let got = cpu.forward(fx.h, fx.w, &fx.input).expect("run the forward pass");
    let (worst, at, mean) = fx.compare(&got);
    let (y, x, c) = fx.locate(at);
    eprintln!("cpu: worst |diff| {worst:.3e} at (y {y}, x {x}, ch {c}), mean |diff| {mean:.3e}");

    // fp32 accumulation order differs between the reference and this engine - the
    // convs alone reorder thousands of additions per output pixel - so exact
    // equality is not the bar. 1e-4 on a [0,1] image is well below the difference a
    // wrong bias-order or a transposed gather would produce (those are 1e-2 or
    // larger), and it is tight enough to catch a substituted activation or a
    // missing residual.
    assert!(worst < 1e-4, "worst |diff| {worst} at (y {y}, x {x}, ch {c}) exceeds 1e-4");
}

/// The SCALE-3 forward, against a fixture made from the reference's own scale-3
/// branch. See `x3_model_path` for how the checkpoint and the fixture are produced.
///
/// This is the only test that reaches `Weights::up_blocks`' second branch: a single
/// 64 -> 576 convolution and a 3x pixel shuffle, on a head whose output is 3x rather
/// than 2^n. The engine REJECTED scale 3 outright before this (an earlier version of
/// `validate` returned an error for any non-power-of-two), so this is a path that
/// used to be a documented limitation and is now a checked-in guarantee.
#[test]
fn cpu_matches_the_reference_on_the_48x48_x3_fixture() {
    let Some(model) = x3_model_path() else {
        eprintln!("SKIP: no scale-3 model (set HAT_X3_MODEL; see x3_model_path for how \
                   it is built)");
        return;
    };
    let fx = Fixture::load("tests/data/hats_x3_48x48.bin").expect("load the golden fixture");
    assert_eq!(fx.scale, 3, "the x3 fixture must record scale 3");
    let wt = Weights::load(&model).expect("load the converted checkpoint");
    assert_eq!(wt.scale, fx.scale, "the checkpoint's scale and the fixture's disagree");
    // The head's shape, which is the whole point: ONE block, widening 64 -> 576.
    assert_eq!(wt.up_blocks(), vec![(3, 9 * wt.head_feat)],
               "a scale-3 checkpoint must describe one 3x block of 9*num_feat");

    let mut cpu = Cpu::new(&wt).expect("build the CPU backend");
    let got = cpu.forward(fx.h, fx.w, &fx.input).expect("run the forward pass");
    let (worst, at, mean) = fx.compare(&got);
    let (y, x, c) = fx.locate(at);
    eprintln!("cpu x3: worst |diff| {worst:.3e} at (y {y}, x {x}, ch {c}), mean {mean:.3e}");
    assert!(worst < 1e-4, "worst |diff| {worst} at (y {y}, x {x}, ch {c}) exceeds 1e-4");
}

/// The engine's own index maps versus the checkpoint's, which is a check on the
/// geometry conventions (window order, the shift's sign, the unfold's C-major
/// channel order) independent of any arithmetic.
#[test]
fn the_index_maps_agree_with_the_checkpoints_own_buffers() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no model (set HAT_MODEL to a converted checkpoint)");
        return;
    };
    let wt = Weights::load(&model).expect("load the converted checkpoint");
    // A plan at the fixture's size, so the shift mask and the maps are the ones a
    // real forward would use.
    let plan = hat::plan::Plan::new(48, 48, wt.window, wt.embed);

    let sa = wt.i64("relative_position_index_SA");
    let ours_sa = hat::plan::rpi_sa(plan.win);
    assert_eq!(sa.len(), ours_sa.len(), "rpi_sa length");
    let bad = (0..sa.len()).find(|&i| sa[i] as i64 != ours_sa[i] as i64);
    assert!(bad.is_none(), "rpi_sa differs at {bad:?}: checkpoint {} vs engine {}", sa[bad.unwrap_or(0)], ours_sa[bad.unwrap_or(0)]);

    // rpi_oca is the one map whose checkpoint buffer contains NEGATIVE entries: the
    // reference gathers `relative_position_bias_table[rpi.view(-1)]` and relies on
    // PyTorch's negative-index wrapping. The engine's map is u32 and stores the
    // folded row `rem_euclid(table_rows)`, which selects exactly the row PyTorch
    // would - so the comparison has to fold the checkpoint's entry the same way.
    let oca = wt.i64("relative_position_index_OCA");
    let ours_oca = hat::plan::rpi_oca(plan.win, plan.owin);
    assert_eq!(oca.len(), ours_oca.len(), "rpi_oca length");
    let rows = ((plan.win + plan.owin - 1) * (plan.win + plan.owin - 1)) as i64;
    assert!(oca.iter().any(|&v| v < 0), "the checkpoint's rpi_oca is expected to contain negatives");
    let bad = (0..oca.len()).find(|&i| oca[i].rem_euclid(rows) as u32 != ours_oca[i]);
    assert!(
        bad.is_none(),
        "rpi_oca differs at {bad:?}: checkpoint {} (folded to {}) vs engine {}",
        oca[bad.unwrap_or(0)],
        oca[bad.unwrap_or(0)].rem_euclid(rows),
        ours_oca[bad.unwrap_or(0)]
    );
}
