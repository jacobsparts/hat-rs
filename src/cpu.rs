//! The CPU backend: the whole network in Rust, parallelised with rayon.
//!
//! THIS IS THE PATH FOR A MACHINE WITH NO USABLE GPU, so it is written to be fast
//! in its own right rather than as a reference for the CUDA one. Three decisions
//! follow from that, and all three are about the SHAPE of the parallelism rather
//! than about arithmetic:
//!
//! 1. PARALLELISM IS OVER OUTPUT PIXELS, NOT OVER THE REDUCTION. A 3x3 convolution
//!    of 144 channels into 144 is 144 multiply-adds per output value; splitting
//!    those over threads costs a per-thread accumulator and a tree to combine them,
//!    and the combine is a full extra pass over the output. Splitting over output
//!    values instead costs nothing: each output is written once, by one thread, and
//!    the reduction inside it runs in registers in the order the reference uses.
//!    The exception is the token matmul, where the reduction is over 144 elements
//!    and the output is large enough on its own - see `linear`.
//! 2. THE CHANNEL AXIS IS INNERMOST IN THE WEIGHTS, which is the layout the
//!    checkpoints are in (`[out][in][3][3]` for a conv, `[out][in]` for a linear).
//!    Reading it in that order makes the accumulation order exactly the
//!    reference's - c ascending - which is what makes this backend's output
//!    comparable with the reference at all: a different summation order is a
//!    different float, and the parity tolerance is set by the largest error that
//!    the reference's own arithmetic can produce.
//! 3. NO ALLOCATION IN THE INNER LOOPS. Every intermediate has a buffer sized by
//!    the plan, and the block loop reuses them. The whole-run allocation is what
//!    `--mem` reports and what the tiler budgets against, so a backend that
//!    allocated per block would make that number a lie.
//!
//! The graph is the reference's, term for term, with the two window operators
//! expressed through `plan.rs`'s index maps:
//!
//!   x = conv_first(input)
//!   for each stage: x = x + conv3x3(residual_group(x))     [RHAG's residual path]
//!   x = conv_after_body(x) + x
//!   x = conv_last(upsample(leaky_relu(conv_before_upsample(x))))
//!
//! and `residual_group` is `depth` HABs followed by the stage's OCAB, exactly as
//! `AttenBlocks.forward` does it.
use rayon::prelude::*;

use crate::backend::{finish, Backend, Pre};
use crate::plan::{unfold_index, window_index, Plan};
use crate::weights::Weights;

/// The scratch set sized by the plan, so the block loop allocates nothing.
///
/// The FIELD LIST IS THE GRAPH'S WORKING SET, and it is worth reading as such: a
/// HAB needs norm1's output as both a plane (for the 3x3 convolutional branch) and
/// as windowed tokens (for the attention), so `plane`/`plane2` carry the two plane
/// layouts and `tok` the token one; the attention needs its own q, k, v and output
/// at window-token shapes, and the OCAB needs the same at its own key count. Every
/// buffer here is allocated once per plan and refilled in place, which is what makes
/// `per_pixel_floats` a real budget rather than a guess.
struct Acts {
    /// [c][hp][wp] - the plane being worked on (NCHW, the layout the convs want).
    plane: Vec<f32>,
    /// [c][hp][wp] - the second plane: `norm1`'s output while `plane` holds the
    /// residual, and `conv_x` while `plane` holds the attention's scatter.
    plane2: Vec<f32>,
    /// [tokens][c] - the token view (NHWC), which is the layout every matmul wants.
    tok: Vec<f32>,
    /// [tokens][c] - a second token buffer.
    tok2: Vec<f32>,
    /// [nw][wq][c] - windowed tokens: queries, and the attention's output.
    win: Vec<f32>,
    win_out: Vec<f32>,
    /// [nw][wq][c] - the local attention's keys and values.
    kw: Vec<f32>,
    vw: Vec<f32>,
    /// [nw][wk][c] - the OCAB's keys and values, after the unfold.
    okv: Vec<f32>,
    ovv: Vec<f32>,
    /// [nw][wq][c] - the OCAB's windowed queries, and its attention output.
    oq: Vec<f32>,
    oout: Vec<f32>,
    /// [nw*wq*3c] - the qkv projection's output for one attention.
    qkv: Vec<f32>,
    /// [tokens][mlp_ratio*c] - one block's MLP hidden state.
    hidden: Vec<f32>,
    /// The index maps, built once per plan.
    wi_shift: Vec<u32>,
    wi_plain: Vec<u32>,
    k_idx: Vec<u32>,
    k_keep: Vec<u8>,
    /// The relative-position index maps (per plan, not per block: they are
    /// geometry).
    rpi_sa: Vec<u32>,
    rpi_oca: Vec<u32>,
    /// [nw][wq][wq] - the shifted-window mask, or empty when the mask is all zeros.
    mask: Vec<f32>,
    /// [c][hp][wp] - the CAB's first 3x3 conv's output, held across the GELU and the
    /// second conv.
    hidden_plane: Vec<f32>,
    /// [c][hp][wp] - the CAB's second conv's output, held while the channel
    /// attention's gate is computed from it.
    mid_plane: Vec<f32>,
    /// [c] - the channel attention's pooled mean, one value per channel. Not a plane
    /// and never was: `AdaptiveAvgPool2d(1)`.
    pooled: Vec<f32>,
    /// [squeezed] - the channel attention's hidden width.
    seq_hidden: Vec<f32>,
    /// [c] - the channel attention's sigmoid gate.
    gate: Vec<f32>,
    /// [c][hp][wp] and [2c][hp][wp] - the OCAB's query projection and its key/value
    /// projection in the reference's `cat(k, v)` order.
    qplane: Vec<f32>,
    kvplane: Vec<f32>,
    /// [nw*wq*c] - the OCAB's projected attention output.
    proj: Vec<f32>,
    /// [c][hp][wp] - CONV_FIRST's output, held for the whole forward. The reference
    /// rebinds `x = conv_first(x)` and then adds THAT back after `conv_after_body`, so
    /// this has to survive every stage. It lived in `tok2` - which is scratch that
    /// every HAB's attention projection and MLP write into - so the head added a
    /// block's MLP output instead (measured: `conv_before_upsample` came out with
    /// std 32.4 against the reference's 1.2521, a factor of ~26).
    body: Vec<f32>,
    /// [c][hp][wp] - the RHAG's residual base, i.e. the STAGE's input. It cannot live
    /// in any of the scratch planes (`plane2` holds inter-stage temporaries and
    /// `qplane` is the HAB's and the OCAB's windowed-query scratch), which is exactly
    /// the bug this buffer exists to prevent: saving the stage input in `qplane` made
    /// `+ x` add another block's attention queries.
    resi: Vec<f32>,
    /// [3][oh][ow] - the output plane, in the PADDED geometry (`hp*scale` by
    /// `wp*scale`), which `Backend::forward` then crops and denormalises. Held here so
    /// a second forward pass on the same plan reuses the allocation.
    out_plane: Vec<f32>,
}

impl Acts {
    /// The total length of every buffer this holds, which is what the plan's
    /// memory footprint is made of. Kept as one place so a buffer added to the
    /// struct cannot be forgotten here - it is the sum of the buffers listed below
    /// rather than a second list.
    pub fn footprint(&self) -> u64 {
        let v = |x: &Vec<f32>| x.capacity() as u64;
        let u = |x: &Vec<u32>| x.capacity() as u64;
        let b = |x: &Vec<u8>| x.capacity() as u64;
        v(&self.plane) + v(&self.plane2) + v(&self.tok) + v(&self.tok2) + v(&self.win)
            + v(&self.win_out) + v(&self.kw) + v(&self.vw) + v(&self.okv) + v(&self.ovv)
            + v(&self.oq) + v(&self.oout) + v(&self.qkv) + v(&self.hidden)
            + v(&self.mask) + v(&self.hidden_plane) + v(&self.mid_plane) + v(&self.pooled)
            + v(&self.seq_hidden) + v(&self.gate) + v(&self.qplane) + v(&self.kvplane)
            + v(&self.proj) + v(&self.body) + v(&self.resi) + v(&self.out_plane)
            + u(&self.wi_shift) + u(&self.wi_plain) + u(&self.k_idx) + b(&self.k_keep)
            + u(&self.rpi_sa) + u(&self.rpi_oca)
    }

    fn new(plan: &Plan, mlp_ratio: usize) -> Acts {
        use crate::plan::{rpi_oca, rpi_sa};
        let c = plan.c;
        let tok = plan.tokens() * c;
        let (wq, wk, nw) = (plan.wq(), plan.wk(), plan.nw());
        let mask = crate::plan::shift_mask(plan);
        Acts {
            plane: vec![0.0; tok],
            plane2: vec![0.0; tok],
            tok: vec![0.0; tok],
            tok2: vec![0.0; tok],
            win: vec![0.0; nw * wq * c],
            win_out: vec![0.0; nw * wq * c],
            kw: vec![0.0; nw * wq * c],
            vw: vec![0.0; nw * wq * c],
            okv: vec![0.0; nw * wk * c],
            ovv: vec![0.0; nw * wk * c],
            oq: vec![0.0; nw * wq * c],
            oout: vec![0.0; nw * wq * c],
            qkv: vec![0.0; nw * wq * 3 * c],
            hidden: vec![0.0; plan.tokens() * mlp_ratio * c],
            wi_shift: window_index(plan, plan.shift),
            wi_plain: window_index(plan, 0),
            k_idx: Vec::new(),
            k_keep: Vec::new(),
            rpi_sa: rpi_sa(plan.win),
            rpi_oca: rpi_oca(plan.win, plan.owin),
            mask,
            // The CAB's plane temporaries. `hidden_plane` holds the first 3x3 conv's
            // output, whose width is `compressed` (c//compress_ratio, smaller than c
            // for every released model), so a c-channel plane covers it.
            hidden_plane: vec![0.0; tok],
            mid_plane: vec![0.0; tok],
            pooled: vec![0.0; c],
            seq_hidden: vec![0.0; c],
            gate: vec![0.0; c],
            qplane: vec![0.0; tok],
            resi: vec![0.0; tok],
            body: vec![0.0; tok],
            kvplane: vec![0.0; 2 * tok],
            proj: vec![0.0; nw * wq * c],
            out_plane: Vec::new(),
        }
    }

    /// The OCAB's unfold maps, built on first use because they are the largest maps
    /// (169 entries per window, against 256 for the local ones) and only the OCAB
    /// needs them.
    fn unfold_maps(&mut self, plan: &Plan) {
        if self.k_idx.is_empty() {
            let (i, k) = unfold_index(plan);
            self.k_idx = i;
            self.k_keep = k;
        }
    }
}

/// The backend state: the weights and the buffers for the plan the last `forward`
/// used.
pub struct Cpu<'a> {
    wt: &'a Weights,
    acts: Option<Acts>,
    plan: Option<Plan>,
}

impl<'a> Cpu<'a> {
    pub fn new(wt: &'a Weights) -> Result<Cpu<'a>, String> {
        Ok(Cpu { wt, acts: None, plan: None })
    }

    /// The floats the buffers from the last forward ACTUALLY hold, summed over the
    /// allocation. `footprint_floats` is a derivation from the buffer lists and this
    /// is the measurement, so a test can hold the two against each other and the
    /// derivation cannot quietly drift away from the buffers it describes.
    ///
    /// It counts `Acts` only: `head` allocates its own temporaries and frees them, and
    /// `footprint_floats` accounts for those separately, so the two numbers differ by
    /// the head's term by construction.
    pub fn acts_floats(&self) -> u64 {
        match &self.acts {
            Some(a) => a.footprint(),
            None => 0,
        }
    }

    /// The floats one forward pass of this plan allocates, and the part of it that
    /// does NOT shrink with the tile. `--mem` reports the first and `auto_tile`
    /// solves for the tile size that fits a budget.
    ///
    /// It is derived from the buffer lists rather than measured by allocating, so it
    /// can be asked about a tile size before that tile exists - and it is checked
    /// against a real `Acts` at a couple of sizes in `tests/tiling.rs`, so the
    /// derivation cannot drift away from the buffers it describes.
    pub fn footprint_floats(wt: &Weights, plan: &Plan) -> u64 {
        let c = wt.embed as u64;
        let tok = plan.tokens() as u64 * c;
        let nw = plan.nw() as u64;
        let (wq, wk, win) = (plan.wq() as u64, plan.wk() as u64, plan.win as u64);
        // PLANES, in `Acts`: plane, plane2, tok, tok2, body, resi, hidden_plane,
        // mid_plane, qplane, kvplane - ten c-channel planes.
        let planes = 10 * tok;
        // The MLP's hidden state, `[tokens][mlp_ratio * c]`.
        let mlp = tok * wt.mlp_ratio as u64;
        // Window tokens: win, win_out, kw, vw are each `nw * wq * c` and
        // `nw * wq == tokens`; oq, oout likewise; okv/ovv are `nw * wk * c` (the
        // OCAB's larger, strided windows) and the qkv projection is three c-blocks of
        // the queried tokens.
        let windowed = 4 * tok + 2 * tok + 2 * nw * wk * c + 3 * tok + tok;
        // The head's own temporaries, which are NOT in `Acts`. `head` holds THREE
        // planes at once - `cur` (the current feature plane, `feat` channels), `wide`
        // (what the block's convolution writes, `widened` channels) and `next` (the
        // shuffled result, `feat` channels at `r` times the resolution). With the
        // block's input resolution `R` in units of the padded plane its cost is
        // therefore `(feat + widened + feat * r^2) * hw * R^2`, and the peak is the
        // MAXIMUM over the blocks - for a x4 model that is the SECOND octave
        // (`R = 2`), not the first.
        //
        // THIS IS WHY IT IS NOT A FORMULA IN `log2(scale)`: a scale-3 head is a single
        // block with `widened = 9 * feat`, and the count-based form below-the-fold
        // reading of the old code silently sized a 3x head as if `R` were 1 and the
        // widened plane were `4 * feat`.
        let hw = plan.tokens() as u64; // `tok / c`
        let mut peak = 0u64;
        let mut res = 1u64;
        for (r, widened) in wt.up_blocks() {
            let r = r as u64;
            peak = peak.max((wt.head_feat as u64 + widened as u64 + wt.head_feat as u64 * r * r)
                            * hw * res * res);
            res *= r;
        }
        let head = peak + 3 * (wt.scale as u64).pow(2) * tok / c;
        // The output plane, in the padded geometry at the model's scale.
        let out = 3 * (wt.scale as u64).pow(2) * tok / c;
        // THE INDEX MAPS DO NOT SCALE - except the mask, which is per window. They
        // dominate at small sizes: rpi_oca alone is 590 KB and the mask 2.4 MB at
        // 48x48, against a 5 MB activation set.
        let maps_u32 = 2 * tok + nw * wk + win.pow(4) + wq * wk + nw * win.pow(4);
        let maps = maps_u32 + nw * wk; // k_keep is a byte per key, counted as one float
        planes + mlp + windowed + head + out + maps
    }
}

/// 3x3 convolution, `[c_in][h][w] -> [c_out][h][w]`, zero padding, ReLU applied to
/// the output if `relu`.
///
/// `out` MUST BE EXACTLY `c_out * h * w` LONG. The loop is one chunk per output
/// channel, so a longer buffer makes it read past `bias` and produces an index
/// panic at whichever channel is first out of range. Callers that borrow a shared
/// plane-sized buffer for a narrower result must slice it (see `cab`, which writes
/// 6 channels into a 144-channel plane).
///
/// One thread per output element, the channel reduction in registers in
/// ascending-c order. The three row reads are at `y-1`, `y`, `y+1`; the kernel
/// reads `x-1`, `x`, `x+1` inside each, which costs 3 loads per tap and is why the
/// inner loop is written as three separate `for c` passes rather than one pass over
/// a `c*9` window: the reference's accumulation order is (c, dy, dx) with c
/// OUTERMOST, so the three passes are the only order that matches it.
fn conv3x3(inp: &[f32], out: &mut [f32], w: &[f32], bias: &[f32], ci: usize, co: usize,
           h: usize, wd: usize, relu: bool) {
    let hw = h * wd;
    debug_assert_eq!(out.len(), co * hw, "see the contract above");
    // ONE flat parallel loop over `co * hw` output elements, not a loop over channels
    // with a second parallel loop inside it. The nested form gave rayon `co` outer
    // tasks of `hw` inner ones each, and at 48x48 with 144 channels that is 144 tiny
    // parallel regions per convolution with all their bookkeeping; flattening it lets
    // rayon split the real work finely and keeps every core busy on a small plane.
    // BLOCKED ALONG X, NOT ALONG THE CHANNEL AXIS, AND THE DIFFERENCE IS
    // PARALLELISM. The channel-blocked form this replaces (`par_chunks_mut(BLK*hw)`)
    // creates `co/BLK` tasks, which is 36 for the main 144-channel convolution but
    // TWO for the CAB's `Conv3d(c, c/compress_ratio)` and ONE for `conv_last` - so
    // two convolutions that are 10% of the work were running on two threads of
    // 24. Measured in the per-shape table: ci*9=1296 -> 6 channels at 12.6 GFLOPS and
    // 576 -> 3 at 8.0, against 86.9 for 1296 -> 144.
    //
    // Blocking `BLK` consecutive pixels of one row instead gives `co*hw/BLK` tasks -
    // 147456 at 64x64 - while still giving each task `BLK` independent accumulator
    // chains, and it makes the weight reads SHARED: one load of the nine weights
    // serves all `BLK` pixels.
    //
    // THE ARITHMETIC PER OUTPUT IS UNCHANGED - the same taps in the same c-then-t
    // order - so this is bit-identical to the one-pixel-at-a-time form. What must NOT
    // be done is the obvious restructure (accumulate the nine taps, then add), which
    // reassociates and differs by 7.9e-4, three orders past this backend's 3.9e-6
    // parity.
    debug_assert!(BLK == 1 || hw % BLK == 0, "hw must be a multiple of BLK");
    out.par_chunks_mut(BLK).enumerate().for_each(|(gi, chunk)| {
        let i0 = gi * BLK;
        let o = i0 / hw;
        let p0 = i0 % hw;
        let b = bias[o];
        let kbase = o * ci * 9;
        let (y, x0g) = (p0 / wd, p0 % wd);
        let mut acc = [0.0f32; BLK];
        for a in acc.iter_mut() {
            *a = b;
        }
        // INTERIOR FAST PATH. When the whole block lies inside the row, no tap needs
        // clamping and each of the nine tap reads is BLK CONTIGUOUS floats - which is
        // what lets the loop vectorise at all. This is the same defect the matmul had:
        // the general branch below bounds its inner loop by `chunk.len()`, a length the
        // compiler cannot reason about, so it stays one multiply-add per iteration.
        // Measured at ci=144 co=144 64x64, min of interleaved runs: 15.88 -> 9.96 ms at
        // 24 threads (96.3 -> 153.4 GFLOPS) and roughly 2x single-threaded.
        //
        // THE NINE TAPS GO INTO A PER-CHANNEL SUBTOTAL `s` AND THEN INTO `acc`, which
        // is exactly what the general branch does, so this is BIT-IDENTICAL
        // (`max |diff|` exactly 0). Accumulating each tap straight into `acc` is
        // marginally faster and is NOT the same float - it reassociates
        // `bias + t1 + ... + t9` into `bias + (t1 + ... + t9)` and differs by 1.6e-5,
        // which this backend does not take.
        if x0g >= 1 && x0g + BLK <= wd - 1 {
            for c in 0..ci {
                let p = &inp[c * hw..(c + 1) * hw];
                let k = &w[kbase + c * 9..kbase + c * 9 + 9];
                let mut s = [0.0f32; BLK];
                for dy in 0..3usize {
                    if (dy == 0 && y == 0) || (dy == 2 && y + 1 == h) {
                        continue;
                    }
                    let r = &p[(y + dy - 1) * wd..(y + dy) * wd];
                    for dx in 0..3usize {
                        let kk = k[dy * 3 + dx];
                        let base = x0g - 1 + dx;
                        for j in 0..BLK {
                            s[j] += r[base + j] * kk;
                        }
                    }
                }
                for j in 0..BLK {
                    acc[j] += s[j];
                }
            }
        } else {
        for c in 0..ci {
            let p = &inp[c * hw..(c + 1) * hw];
            let k = &w[kbase + c * 9..kbase + c * 9 + 9];
            for j in 0..chunk.len() {
                let x = x0g + j;
                let y0 = y.wrapping_sub(1);
                let y2 = if y + 1 < h { y + 1 } else { y };
                let y0 = if y == 0 { 0 } else { y0 };
                let x0 = x.wrapping_sub(1);
                let x2 = if x + 1 < wd { x + 1 } else { x };
                let x0 = if x == 0 { 0 } else { x0 };
                // The zero padding is real padding, not replication: for a
                // convolution the reference's `padding=1` contributes nothing at
                // the border, so the terms below are added only where the row and
                // column are inside the plane.
                let mut s = 0.0f32;
                if y > 0 {
                    let r = &p[y0 * wd..y0 * wd + wd];
                    if x > 0 { s += r[x0] * k[0]; }
                    s += r[x] * k[1];
                    if x + 1 < wd { s += r[x2] * k[2]; }
                }
                {
                    let r = &p[y * wd..y * wd + wd];
                    if x > 0 { s += r[x0] * k[3]; }
                    s += r[x] * k[4];
                    if x + 1 < wd { s += r[x2] * k[5]; }
                }
                if y + 1 < h {
                    let r = &p[y2 * wd..y2 * wd + wd];
                    if x > 0 { s += r[x0] * k[6]; }
                    s += r[x] * k[7];
                    if x + 1 < wd { s += r[x2] * k[8]; }
                }
                acc[j] += s;
            }
        }
        }
        for (j, d) in chunk.iter_mut().enumerate() {
            *d = if relu { acc[j].max(0.0) } else { acc[j] };
        }
    });
}

/// How many output channels one `conv3x3` task carries at a time. Four independent
/// multiply-add chains are enough to cover the FMA latency; a larger block adds
/// register pressure and, at these channel counts, no throughput. See `conv3x3`.
/// Note this must divide `hw` AND `wd` (the interior path needs whole blocks inside a
/// row); 8 is both, and is the size the measurements favour.
const BLK: usize = 8;

/// 1x1 convolution, `[c_in][h][w] -> [c_out][h][w]`: `out = w * x + b`.
///
/// A matmul over the channel axis at every pixel, so the reduction is over `c_in`
/// and the order is c ascending.
fn conv1x1(inp: &[f32], out: &mut [f32], w: &[f32], bias: &[f32], ci: usize, co: usize, hw: usize) {
    debug_assert_eq!(out.len(), co * hw);
    out.par_chunks_mut(hw).enumerate().for_each(|(o, dst)| {
        let b = bias[o];
        let k = &w[o * ci..(o + 1) * ci];
        dst.par_iter_mut().enumerate().for_each(|(i, d)| {
            let mut acc = b;
            for c in 0..ci {
                acc += inp[c * hw + i] * k[c];
            }
            *d = acc;
        });
    });
}

/// A token matmul: `[rows][c_in] -> [rows][c_out]`, `out = x * W^T + b`.
///
/// Parallel over ROWS, not over output elements: with 256-1024 tokens in a window
/// and only 144 channels there is not enough work per element for rayon's overhead,
/// and a row is 144 multiply-adds which is worth a task. `W` is `[c_out][c_in]` -
/// the checkpoint's layout - so the reference's `x @ W.t()` is `dot(row, W[o])`.
fn linear(x: &[f32], out: &mut [f32], w: &[f32], bias: &[f32], ci: usize, co: usize, rows: usize) {
    debug_assert_eq!(out.len(), rows * co);
    // AN 8x8 REGISTER TILE OVER (ROWS, OUTPUT CHANNELS) WHOSE INNER BOUNDS ARE
    // COMPILE-TIME CONSTANTS, plus a scalar edge path for the leftover rows and
    // columns. BOTH PARTS OF THAT MATTER, and the second one is what the first
    // attempt got wrong.
    //
    // THE TILE IS FOR VECTORISATION, NOT FOR TRAFFIC. An earlier version of this
    // function tiled the same way but bounded its inner loops by the RUNTIME `nr` and
    // `nc`, and it ran at the same speed as the plain row-at-a-time loop (5.8 ms vs
    // 5.8 ms at 24 threads for 4096x144->432) - a length the `while` form cannot
    // reason about is a length LLVM will not vectorise, so the loop stayed one
    // multiply-add per iteration and the 8x8 tile bought nothing. Splitting the full
    // `RT x CT` case into its own branch, where every bound is a constant, is worth
    // 2.4x: measured on the real shape at 24 threads, min of 30 interleaved runs,
    // 5.53 -> 2.29 ms (92.2 -> 222.9 GFLOPS), and single-threaded 7.7 -> 19.0.
    //
    // THE ACCUMULATION IS UNCHANGED. Each of the 64 accumulators carries its own
    // output channel and adds `c` ascending, exactly as the plain loop did, so every
    // output is BIT-IDENTICAL - `max |diff|` exactly 0 against the row-at-a-time form,
    // in both the full and the edge path. The tile moves the loads, not the adds.
    //
    // Parallel over ROW BLOCKS. At RT = 8 the task count drops 8x from the old
    // parallel-over-rows: a window's 256 rows give 32 tasks and a plane's 4096 give
    // 512, both well above the thread count.
    const RT: usize = 8;
    const CT: usize = 8;
    out.par_chunks_mut(RT * co).enumerate().for_each(|(rb, dst)| {
        let r0 = rb * RT;
        let nfr = dst.len() / co;
        let mut i0 = 0;
        while i0 < nfr {
            let nr = (nfr - i0).min(RT);
            let mut o0 = 0;
            while o0 < co {
                let nc = (co - o0).min(CT);
                let mut acc = [[0.0f32; CT]; RT];
                if nr == RT && nc == CT {
                    // THE FAST PATH: every bound below is a constant.
                    for i in 0..RT {
                        for j in 0..CT {
                            acc[i][j] = bias[o0 + j];
                        }
                    }
                    for c in 0..ci {
                        let mut xv = [0.0f32; RT];
                        for i in 0..RT {
                            xv[i] = x[(r0 + i0 + i) * ci + c];
                        }
                        let mut wv = [0.0f32; CT];
                        for j in 0..CT {
                            wv[j] = w[(o0 + j) * ci + c];
                        }
                        for i in 0..RT {
                            for j in 0..CT {
                                acc[i][j] += xv[i] * wv[j];
                            }
                        }
                    }
                    for i in 0..RT {
                        for j in 0..CT {
                            dst[(i0 + i) * co + o0 + j] = acc[i][j];
                        }
                    }
                } else {
                    // THE EDGE: leftover rows of the last block, or a channel count
                    // that is not a multiple of CT (`conv_last` has 3). Same order,
                    // so the same bits.
                    for i in 0..nr {
                        for j in 0..nc {
                            acc[i][j] = bias[o0 + j];
                        }
                    }
                    for c in 0..ci {
                        let mut xv = [0.0f32; RT];
                        for i in 0..nr {
                            xv[i] = x[(r0 + i0 + i) * ci + c];
                        }
                        let mut wv = [0.0f32; CT];
                        for j in 0..nc {
                            wv[j] = w[(o0 + j) * ci + c];
                        }
                        for i in 0..nr {
                            for j in 0..nc {
                                acc[i][j] += xv[i] * wv[j];
                            }
                        }
                    }
                    for i in 0..nr {
                        for j in 0..nc {
                            dst[(i0 + i) * co + o0 + j] = acc[i][j];
                        }
                    }
                }
                o0 += CT;
            }
            i0 += RT;
        }
    });
}

/// LayerNorm over the CONTIGUOUS axis: `norm(x, weight, bias)` on `[rows][c]`.
///
/// The mean and variance are accumulated in one pass in the reference's order
/// (`x - mean` then squared, summed, divided), which `torch.nn.LayerNorm` does in
/// float32 on the CPU; `var` is BIASED (divided by `c`, not `c-1`), and `eps` is
/// 1e-5, not a tunable.
fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], c: usize, rows: usize, eps: f32) {
    x.par_chunks_mut(c).enumerate().take(rows).for_each(|(_, row)| {
        let mut mean = 0.0f32;
        for v in row.iter() {
            mean += *v;
        }
        mean /= c as f32;
        let mut var = 0.0f32;
        for v in row.iter() {
            let d = *v - mean;
            var += d * d;
        }
        var /= c as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        for i in 0..c {
            row[i] = (row[i] - mean) * rstd * weight[i] + bias[i];
        }
    });
}

/// GELU, the exact `erf` form: `nn.GELU`'s default, NOT the tanh approximation.
#[inline]
fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + erf(v * std::f32::consts::FRAC_1_SQRT_2))
}

/// The complementary error function, to f32 accuracy.
///
/// `nn.GELU` on the CPU calls `erff`, so this has to be `erf` and not a rational
/// approximation of gelu: an approximation is a different function by up to 1e-3,
/// which is 30x the parity tolerance this engine uses. Abramowitz and Stegun
/// 7.1.26 has a maximum absolute error of 1.5e-7, i.e. below the f32 resolution of
/// the result.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * (-x * x).exp();
    sign * y
}

/// The channel attention's global average pool: one value per channel.
///
/// The reference is `nn.AdaptiveAvgPool2d(1)`, whose CPU kernel sums in index order
/// and divides once, so the sum here is a plain ascending loop over the plane (and
/// then over the batch, which is always 1).
fn channel_mean(x: &[f32], out: &mut [f32], c: usize, hw: usize) {
    out.par_iter_mut().enumerate().take(c).for_each(|(ci, o)| {
        let mut acc = 0.0f32;
        let p = &x[ci * hw..(ci + 1) * hw];
        for v in p.iter() {
            acc += *v;
        }
        *o = acc / hw as f32;
    });
}

/// `out[c] = in[c] * gate[c]`, broadcast over the plane.
fn channel_scale(x: &mut [f32], gate: &[f32], c: usize, hw: usize) {
    x.par_chunks_mut(hw).enumerate().take(c).for_each(|(ci, p)| {
        let g = gate[ci];
        for v in p.iter_mut() {
            *v *= g;
        }
    });
}

/// `dst += src * scale`, the block's residual form.
fn add_scaled(dst: &mut [f32], src: &[f32], scale: f32) {
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d += *s * scale);
}

/// Copy a padded plane `[c][hp][wp]` into windowed tokens `[nw][wq][c]` through an
/// index map.
///
/// This is `window_partition` with the shift folded in: `idx` already contains the
/// rolled source position, so the load is a gather and no copy of the rolled plane
/// is ever made. The reference's `permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, n, c)`
/// puts the CHANNEL axis innermost, which is what every matmul downstream wants.
fn gather_windows(plane: &[f32], idx: &[u32], out: &mut [f32], c: usize, win: usize, hw: usize) {
    let wq = win * win;
    out.par_chunks_mut(wq * c).enumerate().for_each(|(wi, dst)| {
        let map = &idx[wi * wq..(wi + 1) * wq];
        for (t, src) in map.iter().enumerate() {
            let s = *src as usize;
            let d = &mut dst[t * c..(t + 1) * c];
            for ch in 0..c {
                d[ch] = plane[ch * hw + s];
            }
        }
    });
}

/// The inverse: windowed tokens back to a padded plane. The same map, read the
/// other way, and each destination element written exactly once because a window
/// partition is a bijection on the plane.
fn scatter_windows(tok: &[f32], idx: &[u32], plane: &mut [f32], c: usize, win: usize, hw: usize) {
    let wq = win * win;
    let nw = idx.len() / wq;
    // NO SAFETY ARGUMENT IS NEEDED AND NO SCRATCH IS USED. Writing the plane through
    // the map is correct because a window partition is a BIJECTION on the padded
    // plane: every destination element is written exactly once and none is left
    // untouched, which `window_index`'s bijectivity test pins. An earlier form handed
    // a raw pointer across the rayon boundary wrapped in `unsafe impl Send/Sync`,
    // justified by window disjointness; this form needs no such argument because it
    // never aliases - one serial pass over the maps, reading `tok`, writing `plane`.
    for wi in 0..nw {
        let map = &idx[wi * wq..(wi + 1) * wq];
        let src = &tok[wi * wq * c..(wi + 1) * wq * c];
        for (t, d) in map.iter().enumerate() {
            let d = *d as usize;
            for ch in 0..c {
                plane[ch * hw + d] = src[t * c + ch];
            }
        }
    }
}

/// The OCAB's key/value gather: the `nn.Unfold(kernel=owin, stride=win, padding=opad)`
/// of a `[2c][hp][wp]` plane into `[nw][wk][c]` keys and values.
///
/// TWO THINGS THIS DOES THAT A PLAIN UNFOLD DOES NOT:
///
/// * it de-interleaves. The unfold's channel order is C-MAJOR over the 2c-channel
///   plane - the first `c` channels are the keys, the next `c` the values, for the
///   whole 13x13 window before either repeats - so a single gather of a `2c`-wide
///   vector would have to be split afterwards. Writing keys and values from the same
///   index pass keeps the two in step and lets the kernels read them as separate
///   contiguous buffers.
/// * it honours `keep`. The reference zeroes the plane's border, and a gather cannot:
///   the position is real to the plane but outside the window's own extent, and the
///   distinction is invisible in the index alone. Zeroing here rather than in the
///   attention kernel is what makes the keys the same values the reference sees, so
///   the softmax is over the same 169 terms with the same zero entries.
fn unfold_kv(plane: &[f32], idx: &[u32], keep: &[u8], k: &mut [f32], v: &mut [f32],
             c: usize, wk: usize, hw: usize) {
    let nw = idx.len() / wk;
    k.par_chunks_mut(wk * c).zip(v.par_chunks_mut(wk * c)).enumerate().take(nw)
        .for_each(|(wi, (kd, vd))| {
            let map = &idx[wi * wk..(wi + 1) * wk];
            let kp = &keep[wi * wk..(wi + 1) * wk];
            for (t, src) in map.iter().enumerate() {
                let s = *src as usize;
                let live = kp[t] != 0;
                let krow = &mut kd[t * c..(t + 1) * c];
                let vrow = &mut vd[t * c..(t + 1) * c];
                for ch in 0..c {
                    if live {
                        krow[ch] = plane[ch * hw + s];
                        vrow[ch] = plane[c * hw + ch * hw + s];
                    } else {
                        krow[ch] = 0.0;
                        vrow[ch] = 0.0;
                    }
                }
            }
        });
}

/// Split a `[rows][3c]` qkv buffer into `[rows][c]` q, k and v.
///
/// The reference reshapes to `(rows, n, 3, heads, d)` and takes `[0]`, `[1]`, `[2]`,
/// so the three blocks are CONTIGUOUS and in that order - `qkv[:, :c]`, `qkv[:, c:2c]`,
/// `qkv[:, 2c:]`. A `[rows][3c]` buffer with the three blocks strided by `c` is what
/// this reads; nothing is transposed, so the head layout is `[n][nH][d]` - one
/// token's heads contiguous - which is exactly what `WindowAttention`'s
/// `permute(2, 0, 3, 1, 4)` produces.
fn split_qkv(qkv: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32], c: usize, rows: usize) {
    q.par_chunks_mut(c).zip(k.par_chunks_mut(c)).zip(v.par_chunks_mut(c))
        .enumerate().take(rows)
        .for_each(|(r, ((qd, kd), vd))| {
            let row = &qkv[r * 3 * c..(r + 1) * 3 * c];
            qd.copy_from_slice(&row[..c]);
            kd.copy_from_slice(&row[c..2 * c]);
            vd.copy_from_slice(&row[2 * c..]);
        });
}

/// The local window attention of one HAB: scores, the relative-position bias, the
/// optional shifted-window mask, the softmax, and the apply - all in one pass per
/// (window, head) pair.
///
/// `q`, `k`, `v` are `[nw][nq][c]`; the bias is `[nH][nq][nq]` if pre-gathered, or
/// `None` with `bias_idx` an index map into `bias_tab` (`[npoints][nH]`); `mask` is
/// the whole plan's `[nw][nq][nq]` or empty; the result is `[nw][nq][c]`.
///
/// ONE PASS PER (window, QUERY) MEANS THE SOFTMAX IS OVER A ROW IN REGISTERS. HAT's
/// window is 16x16 = 256 tokens and its head width is 24, so a score row is 256
/// floats: the row is computed, the max found for numerical stability, the
/// exponentials summed, and the values accumulated - all without writing the 256
/// scores back to memory. The reference computes the same numbers in the same order
/// (matmul, bias add, mask add, softmax, matmul) and this is that sequence with the
/// intermediate kept in registers.
///
/// THE PARALLEL AXIS IS (window, query), NOT window - WHICH IT USED TO BE AND WHICH
/// WAS THE WHOLE COST OF THIS KERNEL. A 48x48 plane has nine windows, so parallelising
/// over windows left fifteen of this machine's twenty-four cores idle in the single
/// most expensive kernel of the forward (measured: attention was 58% of operator time
/// before this change). A row is `heads` (256-key score row + softmax + 24-wide
/// accumulate) units of work - about 74 kFLOP - which is a good rayon task, and there
/// are `nw * wq` of them: 2304 at 48x48, 65536 at 256x256. The row and probability
/// buffers come from `for_each_init`, so each worker keeps one pair for the whole
/// region it owns instead of allocating per window.
///
/// THE MAX SUBTRACTION IS `torch.softmax`'s, and it is not optional: the reference's
/// `nn.Softmax(dim=-1)` subtracts the row max before exponentiating, so a version
/// without it differs from the reference in the last bits AND overflows for large
/// scores.
fn window_attention(q: &[f32], k: &[f32], v: &[f32], bias: Option<&[f32]>, bias_idx: Option<&[u32]>,
                    bias_tab: Option<&[f32]>, mask: &[f32], out: &mut [f32],
                    nw: usize, nq: usize, heads: usize, d: usize, scale: f32, has_mask: bool) {
    let c = heads * d;
    // One output row per (window, query); serial over the heads inside it.
    out.par_chunks_mut(c).enumerate().take(nw * nq).for_each_init(
        || (vec![0.0f32; nq], vec![0.0f32; nq]),
        |(row, prob), (r, orow)| {
        let wi = r / nq;
        let qi = r % nq;
        let qw = &q[wi * nq * c..(wi + 1) * nq * c];
        let kw = &k[wi * nq * c..(wi + 1) * nq * c];
        let vw = &v[wi * nq * c..(wi + 1) * nq * c];
        let mw = if has_mask { Some(&mask[wi * nq * nq..(wi + 1) * nq * nq]) } else { None };
        for hd in 0..heads {
            let bias_row = |qi: usize, ki: usize| -> f32 {
                if let Some(b) = bias {
                    // Pre-gathered: [nH][nq][nq].
                    b[hd * nq * nq + qi * nq + ki]
                } else if let (Some(idx), Some(tab)) = (bias_idx, bias_tab) {
                    // The reference's `table[rpi.view(-1)].view(nq, nk, nH).permute(2,0,1)`:
                    // the index map is [nq][nk] and the table [points][nH].
                    let p = idx[qi * nq + ki] as usize;
                    tab[p * heads + hd]
                } else {
                    0.0
                }
            };
            {
                let qr = &qw[qi * c + hd * d..qi * c + hd * d + d];
                let mut max = f32::NEG_INFINITY;
                for ki in 0..nq {
                    let kr = &kw[ki * c + hd * d..ki * c + hd * d + d];
                    let mut acc = 0.0f32;
                    for i in 0..d {
                        acc += qr[i] * kr[i];
                    }
                    // The reference's order: scale is folded into q BEFORE the
                    // matmul (`q = q * self.scale`), so the product is
                    // (q*scale) . k, not (q.k)*scale. At these magnitudes the two
                    // can differ in the last bit, and the fixtures are compared at
                    // 2e-3, so it is the reference's order that is used.
                    let mut s = acc * scale + bias_row(qi, ki);
                    if let Some(m) = mw {
                        s += m[qi * nq + ki];
                    }
                    row[ki] = s;
                    if s > max {
                        max = s;
                    }
                }
                let mut sum = 0.0f32;
                for ki in 0..nq {
                    let e = (row[ki] - max).exp();
                    prob[ki] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                let dst = &mut orow[hd * d..hd * d + d];
                for i in 0..d {
                    dst[i] = 0.0;
                }
                for ki in 0..nq {
                    let p = prob[ki] * inv;
                    let vr = &vw[ki * c + hd * d..ki * c + hd * d + d];
                    for i in 0..d {
                        dst[i] += p * vr[i];
                    }
                }
            }
        }
    });
}

/// The OCAB's attention: the same structure as the local one with different shapes
/// (`nq = win*win` queries against `nk = owin*owin` keys) and no mask.
///
/// It shares `window_attention`'s body deliberately: the two differ in their bias's
/// shape and in the presence of a mask, and a second copy of the softmax would be a
/// second place for the max subtraction to be forgotten.
fn overlap_attention(q: &[f32], k: &[f32], v: &[f32], bias_idx: &[u32], bias_tab: &[f32],
                     out: &mut [f32], nw: usize, nq: usize, nk: usize, heads: usize,
                     d: usize, scale: f32) {
    let c = heads * d;
    out.par_chunks_mut(nq * c).enumerate().take(nw).for_each(|(wi, od)| {
        let qw = &q[wi * nq * c..(wi + 1) * nq * c];
        let kw = &k[wi * nk * c..(wi + 1) * nk * c];
        let vw = &v[wi * nk * c..(wi + 1) * nk * c];
        let mut row = vec![0.0f32; nk];
        let mut prob = vec![0.0f32; nk];
        for hd in 0..heads {
            for qi in 0..nq {
                let qr = &qw[qi * c + hd * d..qi * c + hd * d + d];
                let mut max = f32::NEG_INFINITY;
                for ki in 0..nk {
                    let kr = &kw[ki * c + hd * d..ki * c + hd * d + d];
                    let mut acc = 0.0f32;
                    for i in 0..d {
                        acc += qr[i] * kr[i];
                    }
                    let p = bias_idx[qi * nk + ki] as usize;
                    let s = acc * scale + bias_tab[p * heads + hd];
                    row[ki] = s;
                    if s > max {
                        max = s;
                    }
                }
                let mut sum = 0.0f32;
                for ki in 0..nk {
                    let e = (row[ki] - max).exp();
                    prob[ki] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                let orow = &mut od[qi * c + hd * d..qi * c + hd * d + d];
                for i in 0..d {
                    orow[i] = 0.0;
                }
                for ki in 0..nk {
                    let p = prob[ki] * inv;
                    let vr = &vw[ki * c + hd * d..ki * c + hd * d + d];
                    for i in 0..d {
                        orow[i] += p * vr[i];
                    }
                }
            }
        }
    });
}
/// The CAB: `ChannelAttention(GELU-conv-conv(conv(x)))`, the reference's
/// `CAB.forward`.
///
/// READ OFF `hat_arch.py` RATHER THAN ASSUMED, because the shape here is easy to get
/// wrong in a way that runs: the two inner convolutions are 3x3 (not 1x1), the
/// activation between them is GELU (not ReLU), and THERE IS NO RESIDUAL inside the
/// CAB - the Sequential's last module is the ChannelAttention, whose result is both
/// the multiplication's right-hand side and the CAB's return value. The converted
/// checkpoint's shapes confirm all three: `cab.0.weight` is `[c//compress, c, 3, 3]`
/// and `cab.2.weight` is `[c, c//compress, 3, 3]`.
///
/// The ChannelAttention is the only place a 1x1 conv appears: it pools the WHOLE
/// plane per channel (AdaptiveAvgPool2d(1)), squeezes to `c // squeeze_factor` with
/// a ReLU between the two 1x1s, and ends in a sigmoid that gates the plane.
///
/// `tmp` and `pooled`/`hidden`/`gate` come from `Acts` rather than being allocated
/// here, because this runs once per HAB and HAT-S has 36 HABs per forward.
fn cab(a: &mut Acts, wt: &Weights, p: &str, c: usize, h: usize, w: usize) {
    let comp = wt.compressed;
    let sq = wt.squeezed;
    let hw = h * w;
    // conv1 (3x3, c -> compressed), then GELU, then conv2 (3x3, back to c).
    // THE SHARED BUFFERS KEEP THEIR FULL SIZE. `tmp` is Acts's c-channel plane but
    // this conv writes only `comp` channels, so what is passed to `conv3x3` is an
    // exactly-sized SLICE (it walks `out` one hw-sized chunk per output channel and
    // would otherwise run past the bias) - and the Vec itself is left at plane size,
    // because Acts hands the same buffer to callers that need all c channels and
    // shrinking it here made one of them panic. Only the slice is narrow.
    let mut tmp = std::mem::take(&mut a.hidden_plane);
    let mut mid = std::mem::take(&mut a.mid_plane);
    conv3x3(&a.plane, &mut tmp[..comp * hw], wt.t(&format!("{p}.cab.0.weight")),
            wt.t(&format!("{p}.cab.0.bias")), c, comp, h, w, false);
    for v in tmp[..comp * hw].iter_mut() {
        *v = gelu(*v);
    }
    conv3x3(&tmp[..comp * hw], &mut mid[..c * hw], wt.t(&format!("{p}.cab.2.weight")),
            wt.t(&format!("{p}.cab.2.bias")), comp, c, h, w, false);
    // The channel attention gates the conv2 output - there is no `x +`: the
    // reference's CAB returns `x * y` from the ChannelAttention alone.
    let mut pooled = std::mem::take(&mut a.pooled);
    channel_mean(&mid, &mut pooled, c, hw);
    let mut seq = std::mem::take(&mut a.seq_hidden);
    // `seq` comes from Acts sized at c, but the squeeze writes only `sq` of them, so
    // the slice must be exactly co*hw - the convs walk `out` one hw-sized chunk per
    // output channel and would otherwise index past the bias.
    conv1x1(&pooled, &mut seq[..sq], wt.t(&format!("{p}.cab.3.attention.1.weight")),
            wt.t(&format!("{p}.cab.3.attention.1.bias")), c, sq, 1);
    for v in seq.iter_mut() {
        if *v < 0.0 { *v = 0.0; }
    }
    let mut gate = std::mem::take(&mut a.gate);
    conv1x1(&seq, &mut gate, wt.t(&format!("{p}.cab.3.attention.3.weight")),
            wt.t(&format!("{p}.cab.3.attention.3.bias")), sq, c, 1);
    for v in gate.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
    channel_scale(&mut mid, &gate, c, hw);
    std::mem::swap(&mut a.plane, &mut mid);   // the CAB's result lands in `plane`
    a.hidden_plane = tmp; a.mid_plane = mid; a.pooled = pooled; a.seq_hidden = seq; a.gate = gate;
}

/// The OCAB: `OverlapCrossAttention.forward`, whose SHAPES AND PROJECTIONS had to be
/// read off the source - the summary this file started from described a different
/// block.
///
/// The reference, in order:
///   x = norm1(x)                      (the caller does this)
///   qkv = self.qkv(x)                 ONE Linear(c, 3c) over the plane
///   q = qkv[0], kv = cat(qkv[1], qkv[2])   q is the projected plane's first c
///                                          channels; k and v are the other two c's,
///                                          CONCATENATED ALONG CHANNELS into a 2c plane
///   kv_windows = self.unfold(kv)      the WHOLE 2c plane is unfolded at once, which
///                                     is why the C-major channel order matters: the
///                                     unfold walks the 2c channels of a 24x24 window
///                                     in order, so the first c are k and the rest v
///   q_windows = window_partition(q)   256 queries per window
///   ... attention over 169 keys ...
///   x = proj(attn) + shortcut         the block's input, not the normed value
///   x = x + mlp(norm2(x))
///
/// `plane2` holds the normed plane on entry and the qkv projection's k,v on exit;
/// the caller owns the residual.
fn ocab(a: &mut Acts, wt: &Weights, plan: &Plan, stage: usize) {
    let c = wt.embed;
    let hw = plan.tokens();
    let (nw, wq, wk) = (plan.nw(), plan.wq(), plan.wk());
    let p = format!("layers.{stage}.residual_group.overlap_attn");
    let scale = 1.0 / (wt.head_dim as f32).sqrt();
    a.unfold_maps(plan);

    // qkv over the TOKEN layout of the normed plane (the reference's Linear acts on
    // [b, h*w, c]), then the three c-blocks are split back out onto the plane
    // layout because the unfold works on channels.
    linear(&a.tok, &mut a.qkv, wt.t(&format!("{p}.qkv.weight")), wt.t(&format!("{p}.qkv.bias")),
           c, 3 * c, hw);

    // q: the first c channels, windowed directly (no unfold).
    // CHANNEL-MAJOR `[c][hw]`, for the same reason as the kv plane below:
    // `gather_windows` indexes a plane as `[ch][hw]`, so a token-major buffer here
    // transposes the queries and every window is wrong (the values are all present,
    // just interleaved per token instead of grouped per channel).
    let mut q_plane = std::mem::take(&mut a.qplane);
    q_plane.resize(hw * c, 0.0);
    {
        let qp = &mut q_plane;
        qp.par_chunks_mut(hw).enumerate().take(c).for_each(|(ch, dst)| {
            for i in 0..hw {
                dst[i] = a.qkv[i * 3 * c + ch];
            }
        });
    }
    gather_windows(&q_plane, &a.wi_plain, &mut a.oq, c, plan.win, hw);
    // kv: the second and third c-blocks, as ONE 2c plane in the reference's
    // cat order (k's c channels then v's).
    // CHANNEL-MAJOR, `[2c][hw]` - NOT token-major. The engine's `a.qkv` is
    // `[hw][3c]`, so this is a transpose of the k and v blocks, and the distinction
    // matters because `unfold_kv` reads the plane as channel-major (it looks for the
    // v channels at `c * hw`, which is what its `c`/`hw` arguments mean). Writing it
    // token-major here put the same values in a different order and every key and
    // value window came out wrong while keeping the same multiset of numbers.
    let mut kv_plane = std::mem::take(&mut a.kvplane);
    kv_plane.resize(hw * 2 * c, 0.0);
    {
        let kvp = &mut kv_plane;
        kvp.par_chunks_mut(hw).enumerate().take(2 * c).for_each(|(ch, dst)| {
            // Channel `ch` of the 2c plane is channel `c + ch` of the qkv row when
            // `ch < c` (k) and channel `2c + (ch - c)` when `ch >= c` (v).
            let src = if ch < c { c + ch } else { 2 * c + (ch - c) };
            for i in 0..hw {
                dst[i] = a.qkv[i * 3 * c + src];
            }
        });
    }
    // NOTE the arguments: `c` is the HALF width of the 2c plane and `hw` its pixel
    // count, which is what lets `unfold_kv` find the values at `c * hw`.

    unfold_kv(&kv_plane, &a.k_idx, &a.k_keep, &mut a.okv, &mut a.ovv, c, wk, hw);

    let q_rows = nw * wq;
    // No second projection: the keys and values ARE the projected plane's channels.
    overlap_attention(&a.oq, &a.okv, &a.ovv, &a.rpi_oca,
                      wt.t(&format!("{p}.relative_position_bias_table")),
                      &mut a.oout, nw, wq, wk, wt.heads, wt.head_dim, scale);
    let mut proj = std::mem::take(&mut a.proj);
    proj.resize(q_rows * c, 0.0);

    linear(&a.oout, &mut proj, wt.t(&format!("{p}.proj.weight")), wt.t(&format!("{p}.proj.bias")),
           c, c, q_rows);

    // `window_reverse` onto whatever plane the caller wants the result in; here
    // `plane` holds the block's input (the caller's residual base).
    scatter_windows(&proj, &a.wi_plain, &mut a.plane2, c, plan.win, hw);

    a.qplane = q_plane; a.kvplane = kv_plane; a.proj = proj;
}

fn to_plane(tok: &[f32], plane: &mut [f32], c: usize, hw: usize) {
    plane.par_chunks_mut(hw).enumerate().take(c).for_each(|(ch, dst)| {
        for i in 0..hw {
            dst[i] = tok[i * c + ch];
        }
    });
}

fn to_tokens(plane: &[f32], tok: &mut [f32], c: usize, hw: usize) {
    tok.par_chunks_mut(c).enumerate().take(hw).for_each(|(i, dst)| {
        for ch in 0..c {
            dst[ch] = plane[ch * hw + i];
        }
    });
}

/// One HAB: `HAB.forward` term for term, on `a.plane` in and out.
///
/// Four steps, in the reference's order:
///   shortcut = plane
///   normed   = norm1(plane)                     (LayerNorm over the channel axis)
///   conv_x   = CAB(normed as NCHW)              (3x3 convs, so it needs the plane)
///   x = shortcut + attn(normed as windows) + conv_x * conv_scale
///   x = x + mlp(norm2(x))
///
/// The order of the two additions into the shortcut is the reference's and is NOT
/// commutable in floating point: `attn_x` is added first and `conv_x * conv_scale`
/// second. The MLP reads the SUM, not the pre-attention value.
///
/// The window operator is chosen by the block's parity: even blocks use the plain
/// windows, odd blocks the shift-and-mask pair. Both index maps are built once per
/// plan, and the shift is folded into the map rather than applied as a cyclic roll of
/// the plane, which is the same arithmetic without a 2-plane round trip.
fn hab(a: &mut Acts, wt: &Weights, plan: &Plan, stage: usize, block: usize) {
    let c = wt.embed;
    let hw = plan.tokens();
    let (h, w) = (plan.hp, plan.wp);
    let (nw, wq) = (plan.nw(), plan.wq());
    let p = format!("layers.{stage}.residual_group.blocks.{block}");
    let shifted = block % 2 == 1;
    let scale = 1.0 / (wt.head_dim as f32).sqrt();

    // shortcut -> plane2, norm1(shortcut) -> tok, and its NCHW view -> plane.
    a.plane2.copy_from_slice(&a.plane);
    // A RELAYOUT, NOT A COPY. `a.plane` is channel-major `[c][hw]` and the norm
    // needs the token layout `[hw][c]`; `copy_from_slice` would reinterpret 144
    // consecutive SPATIAL pixel of one channel as one token's 144 channels, so the
    // normalisation would run over the wrong axis and every value downstream of it
    // would be wrong while looking plausible.
    to_tokens(&a.plane, &mut a.tok, c, hw);
    layer_norm(&mut a.tok, wt.t(&format!("{p}.norm1.weight")),
               wt.t(&format!("{p}.norm1.bias")), c, hw, 1e-5);

    // The CAB is a 3x3-conv block, so it runs on the plane layout: `plane` is free
    // now that the shortcut is in plane2, and the normed tokens are copied into it.
    to_plane(&a.tok, &mut a.mid_plane, c, hw);
    std::mem::swap(&mut a.plane, &mut a.mid_plane);
    cab(a, wt, &format!("{p}.conv_block"), c, h, w);

    a.mid_plane.copy_from_slice(&a.plane);         // conv_x
    a.plane.copy_from_slice(&a.plane2);            // back to the shortcut

    // The attention. Its windows are gathered from the NORMED tokens, which are in
    // `a.tok`; `gather_windows` reads a plane, so the normed tokens go back to a
    // plane first - the same `to_plane` result the CAB used, recomputed because the
    // CAB may have overwritten `mid_plane`. `qplane` is free until the OCAB.
    to_plane(&a.tok, &mut a.qplane, c, hw);
    let idx: &[u32] = if shifted { &a.wi_shift } else { &a.wi_plain };
    gather_windows(&a.qplane, idx, &mut a.win, c, plan.win, hw);
    linear(&a.win, &mut a.qkv, wt.t(&format!("{p}.attn.qkv.weight")),
           wt.t(&format!("{p}.attn.qkv.bias")), c, 3 * c, nw * wq);
    // q lands back in `a.win` (the gather's own buffer: q is not read again) and k, v
    // in their own buffers.
    split_qkv(&a.qkv, &mut a.win, &mut a.kw, &mut a.vw, c, nw * wq);
    window_attention(&a.win, &a.kw, &a.vw, None, Some(&a.rpi_sa),
                     Some(wt.t(&format!("{p}.attn.relative_position_bias_table"))),
                     if shifted { &a.mask } else { &[] },
                     &mut a.win_out, nw, wq, wt.heads, wt.head_dim, scale, shifted);
    linear(&a.win_out, &mut a.tok2, wt.t(&format!("{p}.attn.proj.weight")),
           wt.t(&format!("{p}.attn.proj.bias")), c, c, nw * wq);
    scatter_windows(&a.tok2, idx, &mut a.plane, c, plan.win, hw);
    // x = shortcut + attn + conv_x * conv_scale, in that order. `scatter_windows`
    // WRITES `a.plane`, it does not accumulate into it - so when it returns, `plane`
    // holds the attention term ALONE and the shortcut has to be added back here.
    // (It was missing, which dropped the block's own input out of the sum - the
    // whole block's residual connection, silently.) The order of the two adds is the
    // reference's `x = shortcut + drop_path(attn_x) + conv_x * self.conv_scale` and
    // is not commutable in floating point.
    add_scaled(&mut a.plane, &a.plane2, 1.0);
    add_scaled(&mut a.plane, &a.mid_plane, wt.conv_scale);

    // x = x + mlp(norm2(x)).
    to_tokens(&a.plane, &mut a.tok, c, hw);
    layer_norm(&mut a.tok, wt.t(&format!("{p}.norm2.weight")),
               wt.t(&format!("{p}.norm2.bias")), c, hw, 1e-5);

    let hidden = wt.mlp_ratio * c;
    linear(&a.tok, &mut a.hidden, wt.t(&format!("{p}.mlp.fc1.weight")),
           wt.t(&format!("{p}.mlp.fc1.bias")), c, hidden, hw);

    for v in a.hidden.iter_mut() {
        *v = gelu(*v);
    }
    linear(&a.hidden, &mut a.tok2, wt.t(&format!("{p}.mlp.fc2.weight")),
           wt.t(&format!("{p}.mlp.fc2.bias")), hidden, c, hw);

    to_plane(&a.tok2, &mut a.plane2, c, hw);
    add_scaled(&mut a.plane, &a.plane2, 1.0);

}

/// The OCAB as its own block, with the reference's residual: `proj(x) + shortcut`
/// then `x + mlp(norm2(x))`, where the shortcut is the block's INPUT.
fn ocab_block(a: &mut Acts, wt: &Weights, plan: &Plan, stage: usize) {
    let c = wt.embed;
    let hw = plan.tokens();
    let p = format!("layers.{stage}.residual_group.overlap_attn");
    a.plane2.copy_from_slice(&a.plane);            // shortcut
    to_tokens(&a.plane, &mut a.tok, c, hw);        // relayout, NOT a copy
    layer_norm(&mut a.tok, wt.t(&format!("{p}.norm1.weight")),
               wt.t(&format!("{p}.norm1.bias")), c, hw, 1e-5);
    // `ocab` writes the attended, projected plane into `plane` and takes the normed
    // tokens from `tok`; the shortcut is safe in plane2.
    ocab(a, wt, plan, stage);
    add_scaled(&mut a.plane, &a.plane2, 1.0);
    to_tokens(&a.plane, &mut a.tok, c, hw);
    layer_norm(&mut a.tok, wt.t(&format!("{p}.norm2.weight")),
               wt.t(&format!("{p}.norm2.bias")), c, hw, 1e-5);

    let hidden = wt.mlp_ratio * c;
    linear(&a.tok, &mut a.hidden, wt.t(&format!("{p}.mlp.fc1.weight")),
           wt.t(&format!("{p}.mlp.fc1.bias")), c, hidden, hw);
    for v in a.hidden.iter_mut() {
        *v = gelu(*v);
    }
    linear(&a.hidden, &mut a.tok2, wt.t(&format!("{p}.mlp.fc2.weight")),
           wt.t(&format!("{p}.mlp.fc2.bias")), hidden, c, hw);

    to_plane(&a.tok2, &mut a.plane2, c, hw);
    add_scaled(&mut a.plane, &a.plane2, 1.0);

}

/// One RHAG: `patch_embed(conv(patch_unembed(atten_blocks(x)))) + x`, where the
/// embed/unembed pair are pure reshapes (`in_chans=0`, `norm_layer=None`) and the
/// conv is 3x3.
///
/// The `+ x` is the STAGE's residual and is added to the stage's input, which is in
/// `plane2` before the conv overwrites it - so the input is copied to `qplane` first.
fn rhag(a: &mut Acts, wt: &Weights, plan: &Plan, stage: usize) {
    let c = wt.embed;
    let (h, w) = (plan.hp, plan.wp);
    // The RHAG's residual base. It goes in `resi` and NOT in `qplane`/`plane2`:
    // those are scratch that every HAB's attention and the OCAB overwrite, so
    // `+ x` would add whatever the last block left behind (measured: the engine's
    // residual had std 4.62 where the reference's is exactly the stage input, 1.249).
    a.resi.copy_from_slice(&a.plane);
    for b in 0..wt.depths[stage] {
        hab(a, wt, plan, stage, b);
    }
    ocab_block(a, wt, plan, stage);
    // patch_unembed: a no-op reshape here (the plane IS the unembedded form), then
    // the 3x3 conv, then patch_embed back. `a.hidden_plane` is the conv's output.
    conv3x3(&a.plane, &mut a.hidden_plane, wt.t(&format!("layers.{stage}.conv.weight")),
            wt.t(&format!("layers.{stage}.conv.bias")), c, c, h, w, false);
    a.plane.copy_from_slice(&a.hidden_plane);
    add_scaled(&mut a.plane, &a.resi, 1.0);        // + x

}

/// The head: `conv_before_upsample` (3x3 + LeakyReLU 0.01), one 3x3 conv and an
/// r-fold PixelShuffle per upsampling block, `conv_last` (3x3 to 3). The blocks keep
/// their width - `Upsample` carries the same `num_feat` through, which the
/// checkpoint's two `[256, 64, 3, 3]` tensors for a x4 model confirm - and the block
/// list (`[(2, 4*feat)] * n` or `[(3, 9*feat)]`) comes from `Weights::up_blocks()`.
fn head(a: &mut Acts, wt: &Weights, plan: &Plan, y: &mut Vec<f32>) {
    let c = wt.embed;
    let feat = wt.head_feat;
    let (h, w) = (plan.hp, plan.wp);
    let hw = h * w;
    // conv_before_upsample + LeakyReLU(0.01). `mid_plane` is sized for c channels and
    // this needs `feat`, which is 64 for every released model - so it is sized at
    // load time to cover it (`Acts` allocates `max(c, feat)` planes).
    let mut cur = vec![0.0f32; feat * hw];
    let mut wide = vec![0.0f32; 4 * feat * hw];
    let mut next = vec![0.0f32; feat * hw];
    conv3x3(&a.plane, &mut cur, wt.t("conv_before_upsample.0.weight"),
            wt.t("conv_before_upsample.0.bias"), c, feat, h, w, false);

    for v in cur.iter_mut() {
        if *v < 0.0 { *v *= 0.01; }
    }

    // One {conv, shuffle} pair per block, from `Weights::up_blocks()` - the ONE
    // place the reference's two `Upsample` branches (n octaves of `4*num_feat` + a 2x
    // shuffle, or ONE `9*num_feat` + a 3x shuffle) are distinguished. The name index
    // is `2 * block` in both, because `Sequential` numbers the conv 0 and the shuffle
    // 1 whatever the factor is.
    let ch = feat;
    let (mut hh, mut ww) = (h, w);
    for (b, (r, widened)) in wt.up_blocks().iter().enumerate() {
        wide.resize(*widened * hh * ww, 0.0);
        conv3x3(&cur, &mut wide, wt.t(&format!("upsample.{}.weight", 2 * b)),
                wt.t(&format!("upsample.{}.bias", 2 * b)), ch, *widened, hh, ww, false);
        let (nh, nw_) = (hh * r, ww * r);
        next.resize(ch * nh * nw_, 0.0);
        pixel_shuffle(&wide, &mut next, ch, hh, ww, *r);
        std::mem::swap(&mut cur, &mut next);
        hh = nh; ww = nw_;
    }
    let mut out = vec![0.0f32; 3 * hh * ww];
    conv3x3(&cur, &mut out, wt.t("conv_last.weight"), wt.t("conv_last.bias"),
            ch, 3, hh, ww, false);
    *y = out;
}

/// PixelShuffle(r) for `[r*r*c][h][w] -> [c][r*h][r*w]`.
///
/// The reference's `nn.PixelShuffle` reads the input's channel axis as `(c, r, r)`
/// with the SUBSAMPLING axes LAST: input channel `c * r * r + i * r + j` goes to
/// output channel `c` at `(r*y + i, r*x + j)`. The two subsampling factors are in
/// row-major order - rows first, then columns - which is the only order the
/// reference's own `pixel_shuffle` accepts, and getting it backwards produces a
/// plausible-looking image with its blocks transposed.
///
/// `r` IS A RUNTIME ARGUMENT and the two values in the family are 2 (the n-octave
/// `Upsample` chain) and 3 (the single scale-3 block); both come from
/// `Weights::up_blocks()` rather than from the call site.
fn pixel_shuffle(src: &[f32], dst: &mut [f32], c: usize, h: usize, w: usize, r: usize) {
    let (ow, oh) = (w * r, h * r);
    let shw = h * w;
    let dhw = oh * ow;
    dst.par_chunks_mut(dhw).enumerate().take(c).for_each(|(co, d)| {
        for y in 0..h {
            for x in 0..w {
                for i in 0..r {
                    for j in 0..r {
                        d[(r * y + i) * ow + (r * x + j)] =
                            src[(co * r * r + i * r + j) * shw + y * w + x];
                    }
                }
            }
        }
    });
}
/// The whole network on the token/plane working set: `HAT.forward` and
/// `forward_features` together, from the mean-adjusted padded input to the
/// mean-adjusted output plane, both in the padded geometry.
///
/// THE ORDER IS THE REFERENCE'S and it has one trap in it: the body's residual is
/// the CONV_FIRST OUTPUT, not the input image - `x = conv_after_body(forward_features(x)) + x`
/// where `x` was rebound by `x = conv_first(x)`. So conv_first's result has to
/// survive every stage, which is what `a.body` is for here (NOT `tok2`: that is the
/// attention and MLP scratch the blocks reuse, and holding the residual there is a
/// bug this engine already had once).
///
/// The final `self.norm` is a LayerNorm over the channel axis applied AFTER the last
/// RHAG and BEFORE `patch_unembed` - the token layout, not the plane. Its weights
/// exist in every checkpoint (`norm.weight`/`norm.bias`), and leaving it out would
/// still produce an image, just a wrong one: it is the last thing separating the body
/// from the reconstruction head.
fn forward_planes(a: &mut Acts, wt: &Weights, plan: &Plan) -> Result<(), String> {
    let c = wt.embed;
    let (h, w) = (plan.hp, plan.wp);
    let hw = plan.tokens();

    // conv_first reads the 3-CHANNEL padded plane in `a.plane` and widens it to c.
    a.hidden_plane.resize(hw * c, 0.0);
    conv3x3(&a.plane, &mut a.hidden_plane, wt.t("conv_first.weight"), wt.t("conv_first.bias"),
            3, c, h, w, false);
    a.body.resize(hw * c, 0.0);
    a.body.copy_from_slice(&a.hidden_plane);   // the body residual, kept all forward
    a.plane.resize(hw * c, 0.0);
    a.plane.copy_from_slice(&a.hidden_plane);

    // HAT's OWN `patch_embed` norm, which is `PatchEmbed(patch_size=1, norm_layer=...)`:
    // with a patch size of one its `proj` is an Identity, so it is a flatten followed
    // by a LayerNorm over the tokens - and unlike the blocks' `norm1`/`norm2`, its
    // weights are REAL and are in the checkpoint (`patch_embed.norm.weight/bias`).
    // It runs before the first RHAG, on the token layout, which is where the reference
    // calls `self.patch_embed(x)` inside `forward_features`.
    to_tokens(&a.plane, &mut a.tok, c, hw);
    layer_norm(&mut a.tok, wt.t("patch_embed.norm.weight"), wt.t("patch_embed.norm.bias"),
               c, hw, 1e-5);

    to_plane(&a.tok, &mut a.plane, c, hw);

    for stage in 0..wt.depths.len() {
        rhag(a, wt, plan, stage);
    }

    // The final norm on the token layout, then `patch_unembed` (a reshape here) back
    // to the plane.
    to_tokens(&a.plane, &mut a.tok, c, hw);
    layer_norm(&mut a.tok, wt.t("norm.weight"), wt.t("norm.bias"), c, hw, 1e-5);
    to_plane(&a.tok, &mut a.plane, c, hw);

    // conv_after_body + the conv_first residual.
    conv3x3(&a.plane, &mut a.hidden_plane, wt.t("conv_after_body.weight"),
            wt.t("conv_after_body.bias"), c, c, h, w, false);

    a.plane.copy_from_slice(&a.hidden_plane);
    // The residual is `a.body`, which has held conv_first's output since the top of
    // this function - NOT `tok2`, which the blocks reuse as scratch.
    add_scaled(&mut a.plane, &a.body, 1.0);

    // The reconstruction head, which writes the output plane into `y`.
    let mut y = std::mem::take(&mut a.out_plane);
    head(a, wt, plan, &mut y);
    a.out_plane = y;
    Ok(())
}
impl<'a> Backend for Cpu<'a> {
    fn name(&self) -> &'static str {
        "cpu"
    }

    /// The whole CPU path: mean-subtract and pad (`Pre`), the network on the padded
    /// plane, then crop and denormalise (`finish`).
    ///
    /// The plan and the buffer set are cached across calls, because a tiled run
    /// calls this once per tile and reallocating the index maps (which are the
    /// largest structures here after the plane temporaries) per tile would dominate
    /// the smaller tiles. A call with a different input size rebuilds them.
    fn forward(&mut self, h: usize, w: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        let pre = Pre::new(self.wt, h, w)?;
        let plan = pre.plan.clone();
        let adjust = pre.adjust(self.wt, input);
        if self.plan.as_ref() != Some(&plan) {
            // THE GUARD SITS AT THE ALLOCATION, so a tiled run checks the TILE's
            // footprint rather than the image's - which is the case where memory is
            // tight and the case `--mem` exists for. See `memguard`.
            crate::memguard::check_cpu(self.wt, h, w)?;
            self.acts = Some(Acts::new(&plan, self.wt.mlp_ratio));
            self.plan = Some(plan.clone());
        }
        let acts = self.acts.as_mut().expect("just installed");
        // The 3-channel padded input goes into the plane the graph reads from.
        acts.plane.clear();
        acts.plane.extend_from_slice(&adjust);
        forward_planes(acts, self.wt, &plan)?;
        let out = std::mem::take(&mut acts.out_plane);
        let r = finish(self.wt, &plan, &out);

        Ok(r)
    }
}
