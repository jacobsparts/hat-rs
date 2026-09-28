//! The tile loop: how wrong a tile is, how right it can be made, and what it costs.
//!
//! `src/backend.rs` claims a tile differs from a whole-image run only near its edges,
//! because a window attention reaches across a whole window and every 3x3 conv widens
//! the field by one pixel. A claim like that is worth exactly as much as the
//! measurement behind it, so this file measures the difference for several tile and
//! margin combinations and prints them; the numbers in the README come from here.
//!
//! It also pins the two properties the loop cannot get wrong:
//!
//! * with a margin that covers the whole image, every tile IS the whole image, so the
//!   result must equal the untiled one exactly - not approximately - which is what
//!   catches an assembly bug (a transposed core, an off-by-one seam) that a tolerance
//!   would hide;
//! * a tile that covers the whole image is one tile, i.e. `--tile` larger than the
//!   image is not a tile at all.
use hat::backend::Backend;
use hat::cpu::Cpu;
use hat::plan::Plan;
use hat::tile;
use hat::weights::Weights;

fn model_path() -> Option<String> {
    if let Ok(p) = std::env::var("HAT_MODEL") {
        return Some(p);
    }
    let p = "../models/hat-s-x4.safetensors".to_string();
    std::path::Path::new(&p).exists().then_some(p)
}

/// A deterministic input with structure at several scales, because a flat or a smooth
/// image would hide a seam: the difference a tile makes lives at the edges, and an
/// image that is constant there would measure as zero for the wrong reason.
fn synthetic(h: usize, w: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let fy = y as f32 / 7.0 + c as f32;
                let fx = x as f32 / 5.0;
                // Fine detail plus a smooth ramp plus a periodic term, so a window
                // boundary lands on something that changes.
                v[(c * h + y) * w + x] =
                    0.5 + 0.25 * (fy.sin() * fx.cos()) + 0.15 * ((x + y) as f32 * 0.7).sin()
                        + 0.1 * (y as f32 / h as f32);
            }
        }
    }
    v
}

fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "comparing outputs of different lengths");
    let mut worst = 0.0f32;
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        worst = worst.max(d);
        sum += d as f64;
    }
    (worst, (sum / a.len() as f64) as f32)
}

#[test]
fn a_tile_that_covers_the_image_is_one_tile() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no model (set HAT_MODEL to a converted checkpoint)");
        return;
    };
    let wt = Weights::load(&model).expect("load the checkpoint");
    let mut cpu = Cpu::new(&wt).expect("build the backend");
    let (h, w) = (32, 32);
    let img = synthetic(h, w);
    let whole = cpu.forward(h, w, &img).expect("whole-image forward");
    for tile_px in [h, h + 32, 4096] {
        let mut be = Cpu::new(&wt).expect("build the backend");
        let (tiled, info) = tile::forward(&mut be, h, w, &img, tile_px, 0, wt.scale, wt.window)
            .expect("tiled forward");
        assert_eq!(info.tiles, 1, "a {tile_px}px tile on a {h}x{w} image is not a tile");
        let (worst, _) = diff(&whole, &tiled);
        assert_eq!(worst, 0.0, "a single {tile_px}px tile must reproduce the whole-image run bit for bit");
    }
}

#[test]
fn a_margin_that_covers_the_image_reproduces_the_whole_image_exactly() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no model (set HAT_MODEL to a converted checkpoint)");
        return;
    };
    let wt = Weights::load(&model).expect("load the checkpoint");
    let (h, w) = (32, 32);
    let img = synthetic(h, w);
    let whole = Cpu::new(&wt).expect("build").forward(h, w, &img).expect("whole");
    let mut be = Cpu::new(&wt).expect("build");
    // Cores of one window, with a margin wider than the image: every sub-image is
    // the whole image, so every core's arithmetic is the arithmetic the whole-image
    // run did, and the assembly has to put the pieces back exactly where they came
    // from. Any seam error at all shows up as a large difference here.
    let (tiled, info) = tile::forward(&mut be, h, w, &img, wt.window, 3 * wt.window, wt.scale, wt.window)
        .expect("tiled forward");
    assert!(info.tiles > 1, "the test is only meaningful with several tiles");
    let (worst, mean) = diff(&whole, &tiled);
    assert_eq!(worst, 0.0, "a covered margin must be exact; worst {worst}, mean {mean}");
}

#[test]
fn the_tile_margin_bounds_how_wrong_a_tile_is() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no model (set HAT_MODEL to a converted checkpoint)");
        return;
    };
    let wt = Weights::load(&model).expect("load the checkpoint");
    let (h, w) = (64, 64);
    let img = synthetic(h, w);
    let whole = Cpu::new(&wt).expect("build").forward(h, w, &img).expect("whole");
    // THIS IS THE MEASUREMENT the README quotes. A bigger margin is never worse on
    // this metric, and the point of the table is to show where it stops paying: the
    // error falls off as the margin grows past the model's reach, and the number of
    // tiles grows as the cores shrink.
    let mut rows = Vec::new();
    for margin in [0, wt.window, 2 * wt.window, 4 * wt.window] {
        let mut be = Cpu::new(&wt).expect("build");
        let (tiled, info) = tile::forward(&mut be, h, w, &img, 32, margin, wt.scale, wt.window)
            .expect("tiled forward");
        let (worst, mean) = diff(&whole, &tiled);
        rows.push((margin, info.tiles, worst, mean));
    }
    for (margin, tiles, worst, mean) in &rows {
        println!("  margin {margin:2}px: {tiles} tiles, worst {worst:.3e}, mean {mean:.3e}");
    }
    // A no-margin tile is the degenerate case and MUST be visibly wrong: if it is
    // not, this test is measuring nothing (a constant image, a broken comparison, or
    // a tiler that quietly ran the whole image).
    assert!(rows[0].2 > 1e-3, "a marginless tile should differ visibly, got {}", rows[0].2);
    // And a margin of two windows - what the CLI uses - has to bring it down by
    // orders of magnitude. The bound is loose on purpose: this asserts the SHAPE of
    // the result (a margin helps, and by a lot), not a specific number, which is
    // what the printed table is for.
    assert!(rows[2].2 < rows[0].2 / 10.0,
            "2 windows of margin should cut the error by an order of magnitude: {} vs {}",
            rows[0].2, rows[2].2);
}

#[test]
fn auto_tile_returns_a_window_multiple_that_fits_the_budget() {
    // A pure function of the budget, so no checkpoint is needed. The costs are the
    // ones `Cpu::footprint_floats` produces for hat-s at 48x48, divided per pixel:
    // ~1.5 k floats per padded pixel and ~15 M floats of index maps that do NOT
    // shrink with the tile - which is the property that makes a very small budget
    // unsatisfiable rather than merely slow.
    let win = 16;
    let per_pixel = 1500u64;
    let constant = 15_000_000u64;

    // ONE WINDOW ALREADY COSTS 61.5 MB at these costs (16*16*1500*4 bytes of
    // activations plus the 15 M-float constant part), so a 32 MiB budget has no tile
    // at all. `None` is the honest answer here rather than a one-window tile that
    // exceeds the budget it was asked to respect.
    let one_window_bytes = (win * win) as u64 * per_pixel * 4 + constant * 4;
    assert!(one_window_bytes > 32 << 20, "the test's premise: one window costs more than 32 MiB");
    assert_eq!(tile::auto_tile(1024, 1024, win, 32 << 20, per_pixel, constant, 0), None);

    // 1 GiB: (1 GiB - constant) / (per_pixel * 4 bytes) pixels, square-rooted and
    // rounded DOWN to a window multiple.
    for budget in [1u64 << 30, 2 << 30, 4 << 30] {
        let t = tile::auto_tile(1024, 1024, win, budget, per_pixel, constant, 0).expect("fits");
        assert_eq!(t % win, 0, "a tile must be a window multiple, got {t}");
        let (pixels, constant_bytes, per_pixel_bytes) = ((t * t) as u64, constant * 4, per_pixel * 4);
        assert!(pixels * per_pixel_bytes + constant_bytes <= budget,
                "the auto tile ({t}px = {} bytes) does not fit a {budget}-byte budget",
                pixels * per_pixel_bytes + constant_bytes);
        // And it is within one window of the largest that would: rounding DOWN must
        // not be an excuse to be needlessly conservative.
        let up = t + win;
        assert!(up * up > 1024 * 1024 || (up * up) as u64 * per_pixel_bytes + constant_bytes > budget,
                "a {up}px tile would also have fit the budget, so {t} is too small");
    }

    // And it is capped at the image: a tile bigger than the picture is not a tile.
    assert_eq!(tile::auto_tile(1024, 1024, win, 64 << 30, per_pixel, constant, 0), Some(1024));
    assert_eq!(tile::auto_tile(64, 64, win, 64 << 30, per_pixel, constant, 0), Some(64));

    // THE HALO IS INSIDE THE BUDGET, which is the property an earlier version got
    // wrong: it sized the budget on the CORE, so with a 32px margin a 480px core has
    // 512px sub-images and the flag's promise was broken by exactly the halo. What is
    // bounded is the SUB-IMAGE, so `tile + 2*margin` is what must fit.
    let margin = 32usize;
    for budget in [1u64 << 30, 2 << 30, 4 << 30] {
        let t = tile::auto_tile(1024, 1024, win, budget, per_pixel, constant, margin).expect("fits");
        assert_eq!(t % win, 0, "a tile must be a window multiple, got {t}");
        let side = t + 2 * margin; // the forward that actually runs
        let sub = side * side;
        assert!(sub > 1024 * 1024 || (sub as u64) * per_pixel * 4 + constant * 4 <= budget,
                "the {t}px core's {side}px sub-image does not fit a {budget}-byte budget");
        // A halo can only make the tile smaller, never larger.
        let bare = tile::auto_tile(1024, 1024, win, budget, per_pixel, constant, 0).unwrap();
        assert!(t <= bare, "a margin must not increase the tile: {t} > {bare}");
    }
    // At the boundary the halo decides: the tile that a budget just barely affords
    // with no margin is the one that overflows it once the halo is counted.
    let t0 = tile::auto_tile(1024, 1024, win, 1 << 30, per_pixel, constant, 0).unwrap();
    let exact = (t0 * t0) as u64 * per_pixel * 4 + constant * 4;
    let with_halo = tile::auto_tile(1024, 1024, win, exact, per_pixel, constant, margin).unwrap();
    assert!(with_halo < t0,
            "a {t0}px core plus a {margin}px halo does not fit the budget a {t0}px core exactly fills");
}

#[test]
fn the_footprint_derivation_matches_the_buffers_it_describes() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no model (set HAT_MODEL to a converted checkpoint)");
        return;
    };
    let wt = Weights::load(&model).expect("load the checkpoint");
    for (h, w) in [(32, 32), (48, 64)] {
        let mut cpu = Cpu::new(&wt).expect("build");
        let img = synthetic(h, w);
        cpu.forward(h, w, &img).expect("forward");
        let measured = cpu.acts_floats();
        let plan = Plan::new(h, w, wt.window, wt.embed);
        let derived = Cpu::footprint_floats(&wt, &plan);
        assert!(measured > 0, "the footprint of a plan that just ran must not be zero");
        // The derivation counts the head's temporaries as well, which `acts_floats`
        // does not, so the two are not equal - but they are the same order, and the
        // measured set is the larger part of it at these sizes.
        assert!(measured < derived, "measured {measured} should be smaller than derived {derived}");
        assert!(derived < 3 * measured,
                "the derivation is {derived} against a measured {measured}: the buffer list has drifted");
    }
}
