//! A backend, and the tiling the two of them share.
//!
//! The trait exists so that the CLI, the tile loop and the fixture checker are
//! written once: `Cpu` and `Gpu` differ only in where the arithmetic happens, and a
//! second copy of the loop above them would be a second place for the padding
//! convention and the crop to disagree.
//!
//! TILING IS NOT EXACT, AND HOW INEXACT IT IS HAS TO BE MEASURED. HAT's receptive
//! field is not local - a window attention mixes the whole 16x16 window (and, in
//! the shifted blocks, half of the neighbouring window), and the OCAB reads a 13x13
//! neighbourhood of a 16-strided grid, i.e. pixels from BOTH sides of a window
//! boundary - but the blocks also include 3x3 convolutions on the whole plane, so
//! the receptive field grows by one pixel per conv. A tile therefore differs from
//! the whole image within a few pixels of its edge, and only because the
//! reference's padding rule is applied to the TILE. Two consequences the engine has
//! to live with rather than hide:
//!
//! * `--tile` is for images that do not fit in memory, and the tile is derived from
//!   the budget, not chosen to be pretty. `tests/tiling.rs` measures the whole-image
//!   difference for a given tile size and margin so the number in the README is a
//!   measurement, not a claim.
//! * The tile and the margin are rounded to window multiples, so the geometry inside
//!   a tile is the geometry the whole image would have had at that offset - the
//!   windows line up; only the padded margin differs.

use crate::plan::Plan;
use crate::weights::Weights;

pub trait Backend {
    fn name(&self) -> &'static str;

    /// One whole-image forward pass. `input` is [3][h][w] in [0,1] (NOT
    /// mean-adjusted); the result is [3][h*scale][w*scale] in [0,1] after the
    /// denormalisation.
    fn forward(&mut self, h: usize, w: usize, input: &[f32]) -> Result<Vec<f32>, String>;
}

/// The reference's preprocessing: pad to the next window multiple, subtract the
/// mean, scale by img_range - and the reverse at the end.
///
/// HAT's own `forward` does neither the padding nor the crop: it reshapes h into
/// `h/16` window rows and raises if that does not divide. The padding here is what
/// its own tiling script does before calling the network, and it is applied by this
/// engine for every input size, so a window-multiple image is the reference's
/// geometry exactly and the fixtures test that case.
pub struct Pre {
    pub plan: Plan,
    pub h: usize,
    pub w: usize,
}

impl Pre {
    pub fn new(wt: &Weights, h: usize, w: usize) -> Result<Pre, String> {
        if h == 0 || w == 0 {
            return Err(format!("{h}x{w} is empty"));
        }
        Ok(Pre { plan: Plan::new(h, w, wt.window, wt.embed), h, w })
    }

    /// Mean-adjusted input, in the network's units, on the PADDED plane the
    /// backend expects. This is the ONLY place the mean is subtracted, matching the
    /// reference's single `x = (x - self.mean) * self.img_range`.
    ///
    /// The padding is EDGE REPLICATION rather than the reflection SwinIR's tiling
    /// uses, and the difference is a choice rather than a derivation: the reference
    /// has no opinion, because it cannot run on a non-multiple at all. Replication
    /// keeps the padded rows equal to the last real row, which for the convs is the
    /// same thing reflection does at a distance of one; the tests measure the
    /// difference between tiled and whole-image runs so the choice is visible in a
    /// number (`tests/tiling.rs`).
    pub fn adjust(&self, wt: &Weights, img: &[f32]) -> Vec<f32> {
        let (h, w, hp, wp) = (self.h, self.w, self.plan.hp, self.plan.wp);
        let hw = h * w;
        let phw = hp * wp;
        let mut out = vec![0.0f32; 3 * phw];
        for c in 0..3 {
            let m = wt.mean[c];
            let src = &img[c * hw..(c + 1) * hw];
            let dst = &mut out[c * phw..(c + 1) * phw];
            for y in 0..hp {
                let sy = y.min(h - 1);
                for x in 0..wp {
                    let sx = x.min(w - 1);
                    dst[y * wp + x] = (src[sy * w + sx] - m) * wt.img_range;
                }
            }
        }
        out
    }
}

/// The reverse: crop the padded output grid back to `h*scale x w*scale` and add
/// the mean back. The only place the denormalisation happens.
pub fn finish(wt: &Weights, plan: &Plan, planes: &[f32]) -> Vec<f32> {
    let (oh, ow) = (plan.hp * wt.scale, plan.wp * wt.scale);
    let (ch, cw) = (plan.h * wt.scale, plan.w * wt.scale);
    let mut out = vec![0.0f32; 3 * ch * cw];
    for c in 0..3 {
        let m = wt.mean[c];
        let src = &planes[c * oh * ow..(c + 1) * oh * ow];
        let dst = &mut out[c * ch * cw..(c + 1) * ch * cw];
        for y in 0..ch {
            for x in 0..cw {
                dst[y * cw + x] = src[y * ow + x] / wt.img_range + m;
            }
        }
    }
    out
}
