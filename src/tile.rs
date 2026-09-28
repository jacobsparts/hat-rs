//! The tile loop: one image too large for memory, run in window-aligned pieces.
//!
//! WHY A MARGIN, AND WHY THE NUMBER IS A MEASUREMENT. HAT's receptive field is not
//! local. A window attention mixes a whole 16x16 window, the shifted blocks mix half
//! of the neighbouring window on each side, and the OCAB reads a 13x13 (24x24 kernel)
//! neighbourhood of a 16-strided grid - i.e. pixels on BOTH sides of a window
//! boundary. On top of that sit 3x3 convolutions on the whole plane, so the field
//! grows by one pixel per convolution. A tile run on its own therefore differs from
//! the whole image near its edges, and the only honest way to publish a tile size is
//! to measure that difference - `tests/tiling.rs` does, for several margins, so the
//! README's number is a reading rather than a claim.
//!
//! WHAT IS EXACT. The tile and the margin are both multiples of the window, and every
//! core starts at a window multiple, so the window grid inside a tile is the window
//! grid the whole image would have had at that offset: the partition, the shift, the
//! OCAB's unfold and the shift mask are all the same as the whole-image run's. The
//! only difference from the whole-image run is that the pieces of plane outside the
//! tile do not exist, which is a padding question and not a convention question -
//! `Pre::adjust`'s edge replication stands in for them, exactly as it does at the
//! image's own border.
//!
//! MEMORY. Peak usage is one tile's working set (see `Cpu::footprint_floats`) plus
//! the input and the output, so `auto_tile` picks the largest window-multiple tile
//! whose working set fits the budget. A tile is never smaller than one window - below
//! that the padding dominates and the picture is wrong rather than just slower - and
//! never larger than the image, where there is nothing to tile.
use crate::backend::Backend;

/// What a tiled run did, for the CLI to report and for a caller to reason about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tiling {
    /// The core's side, in input pixels, rounded to a window multiple.
    pub tile: usize,
    /// The halo on each side, in input pixels, rounded to a window multiple.
    pub margin: usize,
    /// The scale factor of the model that ran.
    pub scale: usize,
    /// How many tiles were run.
    pub tiles: usize,
    /// The largest single forward's input size, in pixels - the memory driver.
    pub largest: (usize, usize),
}

/// One tiled forward pass. `tile` and `margin` are in INPUT pixels; `margin` is the
/// halo each tile is given on every side that is not the image's own edge.
///
/// The output is `[3][h*scale][w*scale]`, in [0,1], exactly like an untiled
/// `Backend::forward` - the seams are not blended, because blending would hide the
/// seam error rather than leaving it where `tests/tiling.rs` can measure it.
pub fn forward<B: Backend>(
    be: &mut B,
    h: usize,
    w: usize,
    input: &[f32],
    tile: usize,
    margin: usize,
    scale: usize,
    win: usize,
) -> Result<(Vec<f32>, Tiling), String> {
    if h == 0 || w == 0 || input.len() < 3 * h * w {
        return Err(format!("tile: bad input {}x{} with {} floats", h, w, input.len()));
    }
    // Rounded to the window: the core has to start on the whole image's window grid
    // and the halo has to be a whole number of windows for the geometry to line up.
    let step = round_up(tile, win).max(win);
    let margin = round_up(margin, win);
    let (oh, ow) = (h * scale, w * scale);
    let mut out = vec![0.0f32; 3 * oh * ow];
    let tiles_y = h.div_ceil(step);
    let tiles_x = w.div_ceil(step);
    let mut tiles = 0usize;
    let mut largest = (0usize, 0usize);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let y0 = ty * step;
            let x0 = tx * step;
            let y1 = (y0 + step).min(h);
            let x1 = (x0 + step).min(w);
            // The halo, clamped to the image. Both bounds are window multiples (or
            // the image's own edge), so the sub-image's window grid is aligned with
            // the whole image's.
            let sy0 = y0.saturating_sub(margin);
            let sx0 = x0.saturating_sub(margin);
            let sy1 = (y1 + margin).min(h);
            let sx1 = (x1 + margin).min(w);
            let (sh, sw) = (sy1 - sy0, sx1 - sx0);
            largest = (largest.0.max(sh), largest.1.max(sw));
            let sub = crop(input, w, sx0, sy0, sw, sh);
            let planes = be.forward(sh, sw, &sub)?;
            // Copy the core - NOT the halo - into the output, at the core's own
            // position scaled by the model.
            let (dy0, dx0) = ((y0 - sy0) * scale, (x0 - sx0) * scale);
            let (ch, cw) = ((y1 - y0) * scale, (x1 - x0) * scale);
            for c in 0..3 {
                let src = &planes[c * sh * scale * sw * scale..];
                let dst = &mut out[c * oh * ow..];
                for y in 0..ch {
                    let s = (dy0 + y) * sw * scale + dx0;
                    let d = ((y0 * scale) + y) * ow + x0 * scale;
                    dst[d..d + cw].copy_from_slice(&src[s..s + cw]);
                }
            }
            tiles += 1;
        }
    }
    Ok((out, Tiling { tile: step, margin, scale, tiles, largest }))
}

fn round_up(v: usize, to: usize) -> usize {
    if to == 0 { return v; }
    v.div_ceil(to) * to
}

/// A sub-image `[3][sh][sw]` of `[3][h][w]`, copying the three channel planes
/// separately because a tile's rows are not contiguous in the source.
fn crop(input: &[f32], w: usize, x0: usize, y0: usize, sw: usize, sh: usize) -> Vec<f32> {
    let src_h = input.len() / 3 / w;
    let mut out = vec![0.0f32; 3 * sh * sw];
    for c in 0..3 {
        let src = &input[c * src_h * w..(c + 1) * src_h * w];
        let dst = &mut out[c * sh * sw..(c + 1) * sh * sw];
        for y in 0..sh {
            dst[y * sw..(y + 1) * sw].copy_from_slice(&src[(y0 + y) * w + x0..(y0 + y) * w + x0 + sw]);
        }
    }
    out
}

/// The largest tile whose working set fits `budget_bytes`, given a per-pixel cost
/// and a constant part - both from the backend's own accounting, so the choice is
/// made against the same numbers `--mem` reports.
///
/// The answer is a window multiple, at least one window wide, capped at the image
/// (a tile larger than the picture is not a tile). It ROUNDS DOWN: `--mem` is a
/// budget, and a tile that rounds up past it is a tile that does not fit, which is
/// the one thing this function exists to prevent. The cost is that a budget is
/// sometimes not fully used - the next window multiple up may be far away - which is
/// the right trade for a flag whose entire purpose is a memory bound.
///
/// `margin` IS PART OF THE COST. The forward a tile runs is on the core plus its halo,
/// so a budget that covers only the core is not a bound on anything - see the comment
/// in the body, and the test that pins it.
///
/// `None` means not even one window fits. That is a real answer rather than an
/// error: the index maps do not shrink with the tile (`rpi_oca` alone is 590 KB and
/// the mask would be megabytes before `mask_label`), so below some budget no tiling
/// exists at all, and the caller says so instead of running something that will not
/// fit.
pub fn auto_tile(
    h: usize,
    w: usize,
    win: usize,
    budget_bytes: u64,
    per_pixel_floats: u64,
    constant_floats: u64,
    margin: usize,
) -> Option<usize> {
    let bytes_per_pixel = per_pixel_floats.max(1) * 4;
    let constant = constant_floats * 4;
    // THE HALO IS PART OF THE WORKING SET, SO IT IS PART OF THE BUDGET. What a tile
    // actually runs a forward on is the core PLUS `margin` on every side that is not
    // the image's edge, so the buffer set is sized for `side + 2*margin` and not for
    // `side`. Sizing on the core alone is not a conservative approximation - it is
    // simply wrong by the halo, and it made `--mem` unable to bound anything: with a
    // 32px margin a 480px core has 512px sub-images, so `--mem 6000` on a 512x512
    // image chose a tile whose largest forward WAS the whole image and ran out of
    // device memory, which is the one thing the flag promises not to do.
    //
    // The bound stays conservative at the edges, where the halo is clamped to the
    // image and the sub-image is therefore smaller than this assumes. A bound may be
    // loose; it may not be exceeded.
    let halo = 2 * margin as u64;
    // One window has to fit before anything else can be considered.
    let one_window = (win as u64 + halo).pow(2) * bytes_per_pixel + constant;
    if budget_bytes < one_window {
        return None;
    }
    let pixels = (budget_bytes - constant) / bytes_per_pixel;
    // A square sub-image gives the largest side for a given pixel budget; the CORE is
    // that minus the halo on both sides.
    let side = (pixels as f64).sqrt() as usize;
    let side = (side.saturating_sub(halo as usize) / win * win).max(win);
    Some(side.min(h.max(w)))
}
