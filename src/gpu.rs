//! The CUDA backend: the same graph as `cpu.rs`, every op a kernel launch.
//!
//! The two backends were written from the same reference independently, so their
//! agreement is evidence rather than a shared transcription - and `--verify
//! --device gpu` runs the same fixture through both.
//!
//! FOUR THINGS DIFFER FROM THE CPU BACKEND, AND ALL FOUR ARE LAYOUT.
//!
//! 1. THE PROJECTIONS ARE PLANAR 1x1 CONVOLUTIONS, NOT GEMMs. A checkpoint Linear
//!    `[c_out][c_in]` applied to an NCHW plane is `lg_conv1x1_rb`, whose contract is
//!    `out[o][p] = bias[o] + sum_c w[o][c] * in[c][p]` - the plane's `hw` columns are
//!    the independent pixels. `lg_f32_gemm_tiled` is NOT that op despite looking like
//!    it: its activation is read as `x + c * ne0` with `c` the COLUMN, so it is a
//!    Linear on a row-major `[tokens][c_in]` buffer - which is what the WINDOW tokens
//!    need and what `lg_linear` computes. Mixing the two up is silent: a plane read as
//!    tokens sums the wrong axis and every value downstream is plausible and wrong.
//!    THE NORM IS PLANAR TOO: the reference normalises over the last axis of
//!    `[b, h*w, c]`, which on a plane is the channel index, and
//!    `lg_channel_layer_norm` reduces over exactly that.
//! 2. THE WINDOW ATTENTION'S `proj` RUNS AT THE WINDOWS rather than after scattering.
//!    A per-token channel mixing commutes with the spatial rearrangement and the
//!    window tokens are contiguous, so this is the same arithmetic in a cheaper order.
//! 3. THE WINDOWS OF A PLANE COME FROM `lg_window_gather`, whose index map is device
//!    side, so nothing is uploaded or built on the host per block - and ONE gather at
//!    `c = 2c` over a 2c plane reproduces `F.unfold`'s layout exactly, because
//!    unfold's flat row index is `ch * (owin*owin) + pos`, which is channel-major
//!    within a window: a window gather of all 2c channels in order.
//! 4. THE MASK IS DERIVED, NOT STORED. The reference materialises `nw * win^4` mask
//!    floats - 1.07 GB at 1024x1024, the engine's largest allocation - while this
//!    backend stores the LABEL grid the mask is a function of (`nw * win^2` i32, 256x
//!    smaller) and the attention derives the addend with one integer compare per
//!    score.
//!
//! WHAT IS *NOT* DIFFERENT, and is where a "conv block" name invites a mistake: the
//! CAB has NO residual (its channel attention's `x * gate` IS its output), the HAB adds
//! `shortcut + attn` BEFORE `+ conv_x * conv_scale`, and every MLP is
//! `x + fc2(gelu(fc1(norm2(x))))`.
//!
//! MEMORY. Every buffer is allocated once per plan and the block loop reuses them, the
//! same discipline as the CPU backend, so the tiler's budget describes this path too.
use std::collections::HashMap;

use lightgpu::vm::DevBuf;

use crate::backend::{finish, Backend, Pre};
use crate::cuda::{self, Cuda};
use crate::plan::Plan;
use crate::weights::Weights;

/// An int64 relative-position index map from the checkpoint, wrapped into its
/// table's row count and narrowed to i32 - the form the gather kernel wants.
///
/// THE WRAP IS THE SEMANTICS, NOT A FIX-UP: PyTorch indexes the table with the
/// checkpoint's own int64 map, whose entries are NEGATIVE in places (`rpi_oca`'s
/// buffer especially, and `rpi_sa`'s has the opposite sign convention), and a
/// negative Python index counts from the end. `rem_euclid` reproduces that exactly.
/// Every table of a given kind has the same row count - 961 = (2*16-1)^2 for the
/// small-window attention - so one wrapped map serves every block of that kind.
fn wrap_i64(wt: &Weights, idx_name: &str, table: &str, n: usize) -> Vec<i32> {
    let rows = wt.shape(table)[0];
    wt.i64(idx_name).iter().take(n).map(|v| v.rem_euclid(rows as i64) as i32).collect()
}

/// LayerNorm's epsilon: `1e-5` in every HAT config, not a tunable.
const EPS: f32 = 1e-5;
/// `lg_conv3x3_winograd`'s activation codes: 0 none, 1 relu, 2 leaky relu with the
/// slope in `act_p`. There is NO gelu among them, so the CAB's activation is a
/// separate `lg_gelu_erf`.
const ACT_NONE: i32 = 0;
const ACT_LEAKY: i32 = 2;
/// `LeakyReLU(0.01)`, in the head and in the MLP.
const LEAKY: f32 = 0.01;

/// Every buffer one forward needs, sized for the plan and allocated once.
///
/// The role of each buffer is FIXED for the whole forward so no kernel ever reads
/// what another is writing; the comments say which term of the reference each holds.
struct Acts {
    /// The plane the graph walks: the current block's input, then its output.
    plane: DevBuf,
    /// Scratch, and the attention's destination before the sum.
    plane2: DevBuf,
    /// `conv_first`'s output, which the reference adds back AFTER the body's stages -
    /// so it survives every block, unlike any scratch plane.
    body: DevBuf,
    /// The current plane's norm (`norm1` / `norm2`).
    n1: DevBuf,
    /// The stage's input: the RHAG's residual base, which no block may overwrite.
    resi: DevBuf,
    /// The HAB's `conv_x` - the CAB's output, scaled and added at the end.
    conv_x: DevBuf,
    /// The MLP's hidden state, `[mlp_ratio * c][hw]` - the plane `fc1` writes and
    /// `fc2` reads, which is why it is `mlp_ratio` times the width of every other
    /// plane here. Getting this wrong is an out-of-bounds write that surfaces as an
    /// illegal address in a LATER kernel, which is how the first run of this backend
    /// failed.
    mlp: DevBuf,
    /// The fused qkv projection of the OVERLAPPING block, `[3c][hw]`: its first c
    /// channels are the queries' plane and the rest are the 2c key/value plane.
    qkv: DevBuf,
    /// The plans the two attentions read: `[c][hw]` for the overlapping block's q and
    /// `[2c][hw]` for its `cat(k, v)`.
    qp: DevBuf,
    kvp: DevBuf,
    /// The CAB's intermediate, `[c][hw]` (only `compressed` channels are used).
    cab_mid: DevBuf,
    /// The channel attention: the pooled mean, the squeeze, and the gate.
    pooled: DevBuf,
    seqh: DevBuf,
    gate: DevBuf,
    /// Window tokens, `[nw*win*win][c]`: the gather's result, the fused qkv, and the
    /// attention's q, k, v and output.
    win: DevBuf,
    qkvw: DevBuf,
    qw: DevBuf,
    kw: DevBuf,
    vw: DevBuf,
    wout: DevBuf,
    /// The overlapping attention's tokens: the windows of its q plane (`[nt][c]`), the
    /// gathered `[nw*owin*owin][2c]` key/value windows, their k and v halves, and the
    /// attention's output.
    oq: DevBuf,
    okw: DevBuf,
    ovv: DevBuf,
    oout: DevBuf,
    /// The gathered relative-position bias, `[heads][nq][nk]`.
    bias: DevBuf,
    /// The shift mask's LABELS, `[nw][win*win]` i32 (the reference stores the mask
    /// itself, 256x larger) and the overlapping attention's key labels.
    lsa: DevBuf,
    loca: DevBuf,
    /// The relative-position INDEX maps, `[nq][nq]` for the small-window attention
    /// and `[nq][nk]` for the overlapping one, already folded into the tables'
    /// row counts with `rem_euclid`.
    ///
    /// THEY LIVE HERE RATHER THAN BEING REBUILT PER BLOCK because they do not depend
    /// on the weights - only on the window geometry, which the plan fixes. Each
    /// BLOCK's bias table does depend on the block, which is why the gather itself is
    /// per block: see `hab`'s call and the note on `maps`.
    idx_sa: DevBuf,
    idx_oca: DevBuf,
    /// The reconstruction head: `h0` is the current `feat`-wide feature plane (the
    /// blocks read and write it) and `h1` is the widened tensor the shuffle consumes -
    /// `4 * feat` per octave, or `9 * feat` for the single scale-3 block, so it is
    /// sized to the WIDEST block (`Weights::up_blocks`).
    ///
    /// `h0` is at the FINAL resolution (the last shuffle writes there) and `h1` at the
    /// LAST BLOCK'S INPUT resolution, because that is where the last convolution runs.
    /// A block passes its own `h`/`w` to the convolution and every kernel derives its
    /// strides from those, so a buffer allocated at the larger size serves every
    /// block - but sizing `h1` at the final resolution wastes `r_last^2` times its
    /// size, 3 GiB of a 512x512 forward.
    h0: DevBuf,
    h1: DevBuf,
    /// The padded output plane, `[3][hp*scale][wp*scale]`.
    out: DevBuf,
}

/// The float count of every device buffer an `Acts` holds at a given plan.
///
/// ONE LIST, TWO USES: `Acts::new` allocates from it and [`footprint_floats`] sums
/// it for the memory guard, so the number the guard refuses on cannot drift from
/// the number that gets allocated. A second list would be a second opinion, and the
/// two would agree right up until the buffer set changed - which is the one moment
/// the guard has to be right.
struct Sizes {
    plane: usize,
    plane2: usize,
    body: usize,
    n1: usize,
    resi: usize,
    conv_x: usize,
    mlp: usize,
    qkv: usize,
    qp: usize,
    kvp: usize,
    cab_mid: usize,
    pooled: usize,
    seqh: usize,
    gate: usize,
    win: usize,
    qkvw: usize,
    qw: usize,
    kw: usize,
    vw: usize,
    wout: usize,
    oq: usize,
    okw: usize,
    ovv: usize,
    oout: usize,
    bias: usize,
    lsa: usize,
    loca: usize,
    idx_sa: usize,
    idx_oca: usize,
    h0: usize,
    h1: usize,
    out: usize,
}

impl Sizes {
    fn new(wt: &Weights, plan: &Plan) -> Sizes {
        let c = wt.embed;
        let hw = plan.tokens();
        let (nw, wq, wk) = (plan.nw(), plan.wq(), plan.wk());
        let nt = nw * wq;
        let feat = wt.head_feat;
        // THE HEAD'S GEOMETRY COMES FROM `up_blocks()`, not from a log2 of the
        // scale: a 2^n model doubles per octave while a scale-3 model triples once,
        // and this buffer has to match whichever the head is about to write.
        //
        // `h0` IS SIZED AT THE FINAL RESOLUTION AND `h1` IS NOT. The shuffle of the
        // LAST block is what writes the final resolution, and that is `h0`; the
        // convolution that feeds it runs one block earlier, at the resolution going
        // INTO that block - `final / r_last`, which is half the final side for the
        // 2^n chain and a third for scale 3. Sizing `h1` at the final resolution
        // instead over-allocates it by `r_last^2` (4x for x4, measured: 768 MiB at
        // 256x256 and 3 GiB at 512x512), and it is the single largest buffer in the
        // head.
        let mut ph = plan.hp;
        let mut pw = plan.wp;
        let mut widest = 0usize;
        let mut ph_in = plan.hp;
        let mut pw_in = plan.wp;
        for (r, widened) in wt.up_blocks() {
            ph_in = ph;
            pw_in = pw;
            ph *= r;
            pw *= r;
            widest = widest.max(widened);
        }
        Sizes {
            plane: c * hw,
            plane2: c * hw,
            body: c * hw,
            n1: c * hw,
            resi: c * hw,
            conv_x: c * hw,
            mlp: wt.mlp_ratio * c * hw,
            qkv: 3 * c * hw,
            qp: c * hw,
            kvp: 2 * c * hw,
            cab_mid: c * hw,
            pooled: c,
            seqh: c,
            gate: c,
            win: nt * c,
            qkvw: nt * 3 * c,
            qw: nt * c,
            kw: nt * c,
            vw: nt * c,
            wout: nt * c,
            oq: nt * c,
            okw: nw * wk * c,
            ovv: nw * wk * c,
            oout: nt * c,
            // SIZED FOR BOTH ATTENTIONS: the HAB's is `[heads][wq][wq]` and the
            // OCAB's `[heads][wq][wk]`, and `wk = owin^2 = 576 > wq = 256`.
            bias: wt.heads * wq * wk,
            lsa: nw * plan.win * plan.win,
            loca: nw * wk,
            idx_sa: wq * wq,
            idx_oca: wq * wk,
            h0: feat * ph * pw,
            h1: widest * ph_in * pw_in,
            out: 3 * ph * pw,
        }
    }

    /// The float count of the whole set. Counted as floats rather than bytes because
    /// every buffer is f32 or i32, so the two are the same size and `total() * 4` is
    /// the byte figure the guard compares.
    fn total(&self) -> u64 {
        let all = [
            self.plane, self.plane2, self.body, self.n1, self.resi, self.conv_x,
            self.mlp, self.qkv, self.qp, self.kvp, self.cab_mid, self.pooled,
            self.seqh, self.gate, self.win, self.qkvw, self.qw, self.kw, self.vw,
            self.wout, self.oq, self.okw, self.ovv, self.oout, self.bias,
            self.lsa, self.loca, self.idx_sa, self.idx_oca, self.h0, self.h1, self.out,
        ];
        all.iter().map(|n| *n as u64).sum()
    }
}

/// The device bytes a forward at this plan allocates, with NO device access: this is
/// what `memguard` refuses on, and it has to answer before anything is allocated -
/// including before a context exists.
pub fn footprint_floats(wt: &Weights, plan: &Plan) -> u64 {
    Sizes::new(wt, plan).total()
}

impl Acts {
    fn new(wt: &Weights, plan: &Plan) -> Result<Acts, String> {
        let s = Sizes::new(wt, plan);
        let (wq, wk) = (plan.wq(), plan.wk());
        let z = |n: usize| DevBuf::zeros(n * 4);
        Ok(Acts {
            plane: z(s.plane)?,
            plane2: z(s.plane2)?,
            body: z(s.body)?,
            n1: z(s.n1)?,
            resi: z(s.resi)?,
            conv_x: z(s.conv_x)?,
            mlp: z(s.mlp)?,
            qkv: z(s.qkv)?,
            qp: z(s.qp)?,
            kvp: z(s.kvp)?,
            cab_mid: z(s.cab_mid)?,
            pooled: z(s.pooled)?,
            seqh: z(s.seqh)?,
            gate: z(s.gate)?,
            win: z(s.win)?,
            qkvw: z(s.qkvw)?,
            qw: z(s.qw)?,
            kw: z(s.kw)?,
            vw: z(s.vw)?,
            wout: z(s.wout)?,
            oq: z(s.oq)?,
            okw: z(s.okw)?,
            ovv: z(s.ovv)?,
            oout: z(s.oout)?,
            bias: z(s.bias)?,
            lsa: z(s.lsa)?,
            loca: z(s.loca)?,
            // THE INDEX MAPS ARE BUILT HERE, not gathered with the bias. They depend
            // only on the window geometry and the tables' ROW COUNTS, both of which this
            // constructor has, and every block of a given kind shares them; the TABLES
            // differ per block, which is why the gather itself is per block (`hab`).
            idx_sa: cuda::upload_i32(&wrap_i64(wt, "relative_position_index_SA",
                                               "layers.0.residual_group.blocks.0.attn.relative_position_bias_table",
                                               wq * wq))?,
            idx_oca: cuda::upload_i32(&wrap_i64(wt, "relative_position_index_OCA",
                                                "layers.0.residual_group.overlap_attn.relative_position_bias_table",
                                                wq * wk))?,
            h0: z(s.h0)?,
            h1: z(s.h1)?,
            out: z(s.out)?,
        })
    }

    /// The floats the buffers ACTUALLY hold, summed over the allocation - the
    /// measurement [`footprint_floats`] is held against, so that the number the guard
    /// refuses on is the number that exists.
    fn footprint(&self) -> u64 {
        let f = |b: &DevBuf| (b.bytes / 4) as u64;
        [
            &self.plane, &self.plane2, &self.body, &self.n1, &self.resi, &self.conv_x,
            &self.mlp, &self.qkv, &self.qp, &self.kvp, &self.cab_mid, &self.pooled,
            &self.seqh, &self.gate, &self.win, &self.qkvw, &self.qw, &self.kw, &self.vw,
            &self.wout, &self.oq, &self.okw, &self.ovv, &self.oout, &self.bias,
            &self.lsa, &self.loca, &self.idx_sa, &self.idx_oca, &self.h0, &self.h1,
            &self.out,
        ]
        .iter()
        .map(|b| f(b))
        .sum()
    }
}

pub struct Gpu<'a> {
    wt: &'a Weights,
    cuda: Cuda,
    /// Every f32 tensor of the checkpoint, uploaded once. The two int64 index buffers
    /// are not weights and are read on the host instead.
    w: HashMap<String, DevBuf>,
    /// The plan and its buffers, rebuilt when the input size changes - a tiled run
    /// calls `forward` once per tile.
    acts: Option<(Plan, Acts)>,
}

impl<'a> Gpu<'a> {
    pub fn new(wt: &'a Weights) -> Result<Gpu<'a>, String> {
        let cuda = Cuda::new()?;
        cuda::available(&cuda)?;
        crate::memguard::check_gpu_weights(wt)?;
        let mut w = HashMap::new();
        for name in &wt.names {
            if name.ends_with("relative_position_index_SA")
                || name.ends_with("relative_position_index_OCA")
            {
                continue;
            }
            w.insert(name.clone(), DevBuf::from_host(wt.t(name))?);
        }
        Ok(Gpu { wt, cuda, w, acts: None })
    }

    /// A checkpoint tensor on the device. Every name used below is in the file:
    /// `weights::load` validates the required list, so a missing one is a bug.
    fn d(&self, name: &str) -> &DevBuf {
        self.w.get(name).unwrap_or_else(|| panic!("gpu: no weight `{name}`"))
    }

    fn plan(&self) -> &Plan {
        &self.acts.as_ref().expect("acts are installed").0
    }

    // -----------------------------------------------------------------------
    // The blocks
    // -----------------------------------------------------------------------

    /// The CAB: `Conv3d(c -> compressed)`, GELU, `Conv2d(compressed -> c)`, then the
    /// channel attention, and NO residual - the channel attention's `x * gate` IS the
    /// block's output, which the reference's own `ConvBlock.forward` makes easy to
    /// get wrong by reading `x + ...` into it.
    ///
    /// `inp` and `out` are distinct planes at `c` channels, because the gate is
    /// derived from `out` and `lg_channel_scale` reads and writes it in one pass.
    fn cab(&self, a: &Acts, inp: &DevBuf, out: &DevBuf, p: &str,
           h: usize, w: usize) -> Result<(), String> {
        let c = self.wt.embed;
        let comp = self.wt.compressed;
        let sq = self.wt.squeezed;
        let hw = h * w;
        if comp == 1 {
            // The two CAB convolutions are a 3x3 `c -> 1` and a 3x3 `1 -> c`. A
            // winograd tile with one input channel is degenerate (the input transform
            // reduces the 6x6 patch to six independent numbers and the weight transform
            // is then a scalar product), so a checkpoint that compressed to a single
            // channel would be served by the direct kernel rather than this one. No
            // released model does, and the branch exists so the fallback is visible
            // rather than implied.
            return Err(format!("{p}: the CAB's compression ratio is 1, which the \
                                winograd path does not cover"));
        }
        self.cuda.conv3x3(inp, &a.cab_mid, self.d(&format!("{p}.cab.0.weight")),
                          self.d(&format!("{p}.cab.0.bias")), c, comp, h, w, ACT_NONE, 0.0)?;
        self.cuda.gelu(&a.cab_mid, &a.cab_mid, comp * hw)?;
        self.cuda.conv3x3(&a.cab_mid, out, self.d(&format!("{p}.cab.2.weight")),
                          self.d(&format!("{p}.cab.2.bias")), comp, c, h, w, ACT_NONE, 0.0)?;
        // ChannelAttention: the pooled mean, then 1x1 -> squeezed, ReLU, 1x1 -> c,
        // Sigmoid. Both 1x1s have a single pixel of input, which is why they use
        // `lg_conv1x1` and not a GEMM (`gemm_tiled` needs four rows of `ne0`, and
        // `sq` can be 6).
        self.cuda.channel_mean(out, &a.pooled, c, hw)?;
        self.cuda.conv1x1(&a.pooled, &a.seqh,
                          self.d(&format!("{p}.cab.3.attention.1.weight")),
                          self.d(&format!("{p}.cab.3.attention.1.bias")), c, sq, 1)?;
        self.cuda.relu(&a.seqh, &a.seqh, sq)?;
        self.cuda.conv1x1(&a.seqh, &a.gate,
                          self.d(&format!("{p}.cab.3.attention.3.weight")),
                          self.d(&format!("{p}.cab.3.attention.3.bias")), sq, c, 1)?;
        self.cuda.sigmoid(&a.gate, &a.gate, c)?;
        self.cuda.channel_scale(out, &a.gate, out, c, hw)
    }

    /// `x + mlp(norm2(x))`, the tail every block of both kinds ends with. `x` is the
    /// block's plane; `n1` is free scratch by the time this runs.
    fn mlp(&self, a: &Acts, p: &str) -> Result<(), String> {
        let c = self.wt.embed;
        let hid = self.wt.mlp_ratio * c;
        let hw = self.plan().tokens();
        self.cuda.channel_layer_norm(&a.plane, self.d(&format!("{p}.norm2.weight")),
                                     self.d(&format!("{p}.norm2.bias")), &a.n1, c, hw, EPS)?;
        // `fc1`: `[hid][c]` weights, `ne0 = c`, `ne1 = hid`. THE OUTPUT PLANE IS THE
        // `[hid][hw]` THAT `fc2` READS: `fc2`'s weights are `[c][hid]`, so its
        // `ne0 = hid` is the same matrix and its columns are the same pixels - which is
        // why the MLP needs no transpose and no separate hidden layout on either
        // backend.
        self.cuda.conv1x1_rb(&a.n1, &a.mlp, self.d(&format!("{p}.mlp.fc1.weight")),
                             self.d(&format!("{p}.mlp.fc1.bias")), c, hid, hw)?;
        self.cuda.gelu(&a.mlp, &a.mlp, hid * hw)?;
        self.cuda.conv1x1_rb(&a.mlp, &a.plane2, self.d(&format!("{p}.mlp.fc2.weight")),
                             self.d(&format!("{p}.mlp.fc2.bias")), hid, c, hw)?;
        self.cuda.add(&a.plane, &a.plane2, &a.plane, c * hw)
    }

    /// One HAB. Every line is a line of the reference's block, in its order.
    fn hab(&self, a: &Acts, stage: usize, block: usize) -> Result<(), String> {
        let plan = self.plan();
        let c = self.wt.embed;
        let (h, w, hw) = (plan.hp, plan.wp, plan.tokens());
        let (nw, wq, nx) = (plan.nw(), plan.wq(), plan.nx());
        let nt = nw * wq;
        let d = self.wt.head_dim;
        let shifted = block % 2 == 1;
        let shift = if shifted { plan.shift } else { 0 };
        let scale = 1.0 / (d as f32).sqrt();
        let p = format!("layers.{stage}.residual_group.blocks.{block}");

        // `n = self.norm1(x)`, written to `n1` because `plane` is the shortcut and has
        // to survive the whole block. `lg_channel_layer_norm` reduces over the CHANNEL
        // axis, which is the reduction the reference performs on `[b, h*w, c]`.
        self.cuda.channel_layer_norm(&a.plane, self.d(&format!("{p}.norm1.weight")),
                                     self.d(&format!("{p}.norm1.bias")), &a.n1, c, hw, EPS)?;
        // The CAB reads the NORMED plane, so it runs before `n1` is reused.
        self.cab(a, &a.n1, &a.conv_x, &format!("{p}.conv_block"), h, w)?;
        // The attention's windows come from the normed plane, rolled by `-shift` for
        // the odd blocks (the gather folds the roll into its index map).
        self.cuda.window_gather(&a.n1, &a.win, nw, wq, nx, plan.win, h, w, c, shift, 0)?;
        // `qkv = self.qkv(windows)`: ONE Linear(c, 3c) over the tokens.
        self.cuda.linear(&a.win, &a.qkvw, self.d(&format!("{p}.attn.qkv.weight")),
                         self.d(&format!("{p}.attn.qkv.bias")), c, 3 * c, nt)?;
        // The three c-blocks of that one projection, split on the token layout.
        self.cuda.token_range(&a.qw, &a.qkvw, c, 3 * c, 0, nt)?;
        self.cuda.token_range(&a.kw, &a.qkvw, c, 3 * c, c, nt)?;
        self.cuda.token_range(&a.vw, &a.qkvw, c, 3 * c, 2 * c, nt)?;
        // The attention, with the gathered relative-position bias and - for the odd
        // blocks - the shift mask, which the kernel derives from `a.lsa`'s labels.
        // `a.lsa` holds the labels for `plan.shift`, and only the shifted blocks use
        // them, which is why one grid serves every odd block.
        // THE BIAS IS GATHERED HERE, PER BLOCK, FROM THIS BLOCK'S OWN TABLE. See the
        // note in `maps`: one table reused for every block is silent and leaves exactly
        // the first block of each stage correct.
        self.gather_bias(a, &format!("{p}.attn.relative_position_bias_table"))?;
        self.cuda.attention(&a.qw, &a.kw, &a.vw, Some(&a.bias), Some(&a.lsa), &a.wout,
                            nw, nx, wq, wq, self.wt.heads, d, scale, true, shifted)?;
        // `self.proj`, applied AT THE WINDOWS: a per-token channel mixing commutes with
        // the spatial rearrangement, so the cheaper order is also the same arithmetic.
        self.cuda.linear(&a.wout, &a.win, self.d(&format!("{p}.attn.proj.weight")),
                         self.d(&format!("{p}.attn.proj.bias")), c, c, nt)?;
        // AFTER the projection, to match the CPU and the reference's own hook on the
        // `attn` module: `WindowAttention.forward` ends `x = self.proj(x)`.
        // The scatter writes the plane, so the sum is explicit and in the reference's
        // order: `x = shortcut + attn + conv_x * conv_scale`.
        self.cuda.window_scatter(&a.win, &a.plane2, nw, wq, nx, plan.win, h, w, c, shift, 0)?;
        self.cuda.add_scaled(&a.plane, &a.plane2, &a.plane2, 1.0, c * hw)?;
        self.cuda.add_scaled(&a.plane2, &a.conv_x, &a.plane, self.wt.conv_scale, c * hw)?;
        self.mlp(a, &p)
    }

    /// The OCAB - `OverlapCrossAttention.forward`, whose shapes had to be read off the
    /// source and whose ONLY tensors are `norm1`, `norm2`, `qkv`, `proj`, the two MLP
    /// matrices and the bias table (there is no key/value projection and no `down`
    /// layer, which a summary of this block once claimed):
    ///
    ///   q, kv = split(self.qkv(norm1(x)))     ONE Linear(c, 3c) over the plane
    ///   q  = window_partition(q)              256 queries per window
    ///   kv = unfold(cat(k, v))                576 keys per window, from a ZERO pad
    ///   a  = attention(q, kv)                 no mask, no relative-position bias
    ///   x  = x + proj(a)                      the shortcut is the BLOCK's input
    ///   x  = x + mlp(norm2(x))
    fn ocab(&self, a: &Acts, stage: usize) -> Result<(), String> {
        let plan = self.plan();
        let c = self.wt.embed;
        let (h, w, hw) = (plan.hp, plan.wp, plan.tokens());
        let (nw, wq, wk, nx) = (plan.nw(), plan.wq(), plan.wk(), plan.nx());
        let nt = nw * wq;
        let d = self.wt.head_dim;
        let scale = 1.0 / (d as f32).sqrt();
        let p = format!("layers.{stage}.residual_group.overlap_attn");

        self.cuda.channel_layer_norm(&a.plane, self.d(&format!("{p}.norm1.weight")),
                                     self.d(&format!("{p}.norm1.bias")), &a.n1, c, hw, EPS)?;
        // ONE fused GEMM over the normed plane, then its bias: the reference's
        // `qkv = self.qkv(x)` on the plane, which is why the GEMM form applies here and
        // the token form (`lg_linear`) applies to the HAB's windowed qkv.
        self.cuda.conv1x1_rb(&a.n1, &a.qkv, self.d(&format!("{p}.qkv.weight")),
                             self.d(&format!("{p}.qkv.bias")), c, 3 * c, hw)?;
        // `q` is the first c channels, windowed at the SMALL window (no overlap), and
        // `kv` is the other 2c as one plane - `cat(k, v)`, which the unfold then reads
        // with the CHANNEL CONTIGUITY that makes its flat order match a window gather of
        // all 2c channels.
        self.cuda.plane_block(&a.qp, &a.qkv, None, c, 3 * c, 0, hw)?;
        self.cuda.plane_block(&a.kvp, &a.qkv, None, 2 * c, 3 * c, c, hw)?;
        self.cuda.window_gather(&a.qp, &a.oq, nw, wq, nx, plan.win, h, w, c, 0, 0)?;
        // THE OVERLAPPING KEYS AND VALUES ARE AN `nn.Unfold`, NOT A WINDOW GATHER.
        // `nn.Unfold(kernel_size = owin = 24, stride = win = 16, padding = opad = 4)`
        // maps key token `t = i*owin + j` of window `wy` to plane row
        // `wy*win + i - opad`, with out-of-plane samples reading ZERO - only 13 of the
        // 24 rows are real. `lg_window_gather` has NO stride argument: with `win = owin`
        // it tiles the plane in non-overlapping 24-strides, which is a completely
        // different set of positions (and a different number of them: 169 real keys,
        // not 256). This is why the OCAB was the last block still wrong while every HAB
        // matched. `hat_unfold_kv` exists for exactly this shape and writes k and v
        // directly, so no `plane_edges` zeroing is needed either: the padding is
        // resolved by the kernel's own bounds test rather than by wrapping modulo the
        // plane.
        self.cuda.unfold_kv(&a.kvp, &a.okw, &a.ovv, nw, nx, plan.win, plan.owin,
                            plan.opad(), h, w, c)?;
        // A BIAS, BUT NO MASK. The reference adds a relative-position bias here exactly
        // as the HAB does (one `[nq][nk]` table per stage, indexed by `rpi_oca`), so the
        // gather below is per stage. It has no MASK: its zero-padded keys carry `k = 0`
        // and are handled by the bias and the values' own zeros. `hat_oca_label` builds
        // the labels that WOULD mask them, and `--cuda-selftest` checks that the two
        // routes agree - the labels are what this engine would need if a checkpoint ever
        // gave a padding row a large value.
        self.gather_bias_oca(a, &format!("{p}.relative_position_bias_table"))?;
        self.cuda.attention(&a.oq, &a.okw, &a.ovv, Some(&a.bias), None, &a.oout,
                            nw, nx, wq, wk, self.wt.heads, d, scale, true, false)?;
        self.cuda.linear(&a.oout, &a.win, self.d(&format!("{p}.proj.weight")),
                         self.d(&format!("{p}.proj.bias")), c, c, nt)?;
        self.cuda.window_scatter(&a.win, &a.plane2, nw, wq, nx, plan.win, h, w, c, 0, 0)?;
        // `x = x + proj(a)`, where `x` is the block's input - the same buffer the
        // `norm1` above read, which is why the norm wrote to `n1`.
        self.cuda.add(&a.plane, &a.plane2, &a.plane, c * hw)?;
        self.mlp(a, &p)
    }

    /// One RHAG: the stage's blocks, the overlapping block, then the stage's own 3x3
    /// convolution as the residual branch.
    fn rhag(&self, a: &Acts, stage: usize) -> Result<(), String> {
        let plan = self.plan();
        let c = self.wt.embed;
        let (h, w, hw) = (plan.hp, plan.wp, plan.tokens());
        // The residual base is the STAGE's input: `+ x` at the end of the stage sums
        // the convolution's output with this, not with the last block's.
        self.cuda.copy(&a.plane, &a.resi, c * hw)?;
        for b in 0..self.wt.depths[stage] {
            self.hab(a, stage, b)?;
        }
        self.ocab(a, stage)?;
        // `patch_unembed` and `patch_embed` are reshapes at patch size 1, so the
        // residual branch is the stage's 3x3 convolution and nothing else.
        self.cuda.conv3x3(&a.plane, &a.plane2, self.d(&format!("layers.{stage}.conv.weight")),
                          self.d(&format!("layers.{stage}.conv.bias")), c, c, h, w,
                          ACT_NONE, 0.0)?;
        self.cuda.add_scaled(&a.plane2, &a.resi, &a.plane, 1.0, c * hw)?;
        Ok(())
    }

    /// The reconstruction head: `conv_before_upsample` and its LeakyReLU, one 3x3
    /// convolution and an r-fold PixelShuffle per upsampling block, then `conv_last`
    /// to three channels. The result lands in `a.out`.
    ///
    /// THE BLOCKS COME FROM `Weights::up_blocks()`, which is the one place the
    /// reference's two `Upsample` branches are distinguished: `log2(scale)` octaves of
    /// `Conv2d(feat, 4*feat, 3)` + `PixelShuffle(2)`, or a SINGLE
    /// `Conv2d(feat, 9*feat, 3)` + `PixelShuffle(3)` for scale 3. Taking a log2 of the
    /// scale instead would run zero blocks for x3 and write `conv_last` straight over
    /// the unupsampled plane.
    fn head(&self, a: &Acts) -> Result<(), String> {
        let plan = self.plan();
        let c = self.wt.embed;
        let feat = self.wt.head_feat;
        let (mut ph, mut pw) = (plan.hp, plan.wp);
        // `conv_before_upsample` is `Sequential(Conv2d(c, feat, 3), LeakyReLU)`:
        // `feat` channels, NOT `4 * feat`. The widening to `4 * feat` is the FIRST
        // convolution of each octave, whose output is what the shuffle consumes. (The
        // first version of this method wrote `4 * feat` here, which reads past a
        // `feat`-channel weight tensor - an out-of-bounds read that surfaced as an
        // illegal address three kernels later, at the next `lg_sigmoid`.)
        //
        // `feat` = 64 is an exact multiple of the winograd kernel's 16-channel `ocb`
        // block, so nothing here needs a padded weight.
        self.cuda.conv3x3(&a.plane, &a.h0, self.d("conv_before_upsample.0.weight"),
                          self.d("conv_before_upsample.0.bias"), c, feat, ph, pw,
                          ACT_LEAKY, LEAKY)?;
        for (b, (r, widened)) in self.wt.up_blocks().iter().enumerate() {
            // Each block is `Conv2d(feat, widened, 3)` then `PixelShuffle(r)`: the
            // widened tensor goes to `h1` and the shuffle writes the r-times-resolution
            // result back into `h0`, which the next block reads.
            self.cuda.conv3x3(&a.h0, &a.h1, self.d(&format!("upsample.{}.weight", 2 * b)),
                              self.d(&format!("upsample.{}.bias", 2 * b)),
                              feat, *widened, ph, pw, ACT_NONE, 0.0)?;
            self.cuda.pixel_shuffle(&a.h1, &a.h0, feat, ph, pw, *r)?;
            ph *= r;
            pw *= r;
        }
        self.cuda.conv3x3(&a.h0, &a.out, self.d("conv_last.weight"), self.d("conv_last.bias"),
                          feat, 3, ph, pw, ACT_NONE, 0.0)?;
        Ok(())
    }

    /// Materialise `table[rpi]` as the `[heads][nq][nk]` addend the attention reads.
    ///
    /// THE TABLE IS A PER-BLOCK WEIGHT: every HAB and every OCAB has its own, all with
    /// the same shape. `a.idx_sa` is the weight-independent index map (wrapped into the
    /// tables' row count when the acts were built), so this is one gather per block
    /// against that block's own table - which is the whole point of the call being here
    /// rather than once in `maps`.
    fn gather_bias(&self, a: &Acts, table: &str) -> Result<(), String> {
        let plan = self.plan();
        self.cuda.bias_gather(self.d(table), &a.idx_sa, &a.bias,
                              plan.wq(), plan.wq(), self.wt.heads)
    }

    /// The OVERLAPPING attention's bias: `[heads][nq][nk]` with `nk = owin^2 = 576`,
    /// gathered from that stage's own table through `a.idx_oca`.
    ///
    /// THE PADDED KEYS CARRY A BIAS TOO, and the reference is explicit about it: it
    /// builds `relative_position_bias` of shape `[nH, ws*ws, wse*wse]` from the
    /// UNFOLD's full 24x24 key count, so the zero-padded keys get their table entry
    /// added to a score of `0 * scale`. Omitting the bias here (which the GPU did,
    /// on the mistaken reading that "the overlapping attention has no bias") left
    /// q, k, v and the unfold all correct to 2e-6 while the attention output was off
    /// by 2.4e-1.
    fn gather_bias_oca(&self, a: &Acts, table: &str) -> Result<(), String> {
        let plan = self.plan();
        self.cuda.bias_gather(self.d(table), &a.idx_oca, &a.bias,
                              plan.wq(), plan.wk(), self.wt.heads)
    }

    fn maps(&self, a: &Acts) -> Result<(), String> {
        let plan = self.plan();
        let (h, w) = (plan.hp, plan.wp);
        let (nw, nx) = (plan.nw(), plan.nx());
        // THE REAL SHIFT. The mask's three BANDS are `slice(0,-win)`, `slice(-win,
        // -shift)` and `slice(-shift,None)`, so their boundaries depend on the shift -
        // while the label grid itself is partitioned at each window's UNROLLED
        // position, which is why `hat_mask_build` reads its coordinates at shift 0
        // internally. Passing 0 here instead (a fix that was briefly applied) makes
        // the last two bands identical and collapses nine regions to four.
        if plan.shift != 0 {
            self.cuda.mask_build(&a.lsa, nw, nx, plan.win, h, w, plan.shift)?;
        }
        self.cuda.oca_label(&a.loca, nw, nx, plan.win, plan.owin, plan.opad(), h, w)?;
        // The relative-position bias for the overlapping attention is NOT gathered:
        // that attention has no bias (see `ocab`). Every HAB's IS, and it is gathered
        // PER BLOCK rather than once here - a gathered `[heads][nq][nk]` addend belongs
        // to ONE table, and the checkpoint stores 42 of them with identical SHAPES and
        // similar magnitudes. Gathering block 0's once and reusing it left block 0 of
        // stage 0 exact to 1.2e-6 while every later block was wrong by ~2e-2, which is
        // a bug that survives a whole verification run.
        //
        // What is built here is only the part that depends on the PLAN rather than the
        // weights: the two label grids. The index maps are in `Acts::new`, because they
        // need the tables' row counts and nothing else.
        Ok(())
    }

    /// The whole graph on the padded plane, whose input must already be uploaded.
    /// `a.out` holds the padded output on return.
    fn forward_planes(&self, a: &Acts) -> Result<(), String> {
        let plan = self.plan();
        let c = self.wt.embed;
        let (h, w, hw) = (plan.hp, plan.wp, plan.tokens());
        // conv_first: the FIRST op, and its output is the residual added after the
        // BODY - not the input image, which is the reference's own trap
        // (`x = conv_after_body(forward_features(x)) + x` where the `x` on the right is
        // conv_first's output).
        self.cuda.conv3x3(&a.plane, &a.body, self.d("conv_first.weight"),
                          self.d("conv_first.bias"), 3, c, h, w, ACT_NONE, 0.0)?;
        // `forward_features`: patch_embed is a reshape, then HAT's own
        // `patch_embed.norm`, the stages, and the final `self.norm` - both norms over
        // the channel axis.
        // THE NORM READS `a.body`, NOT `a.plane`: `a.plane` still holds the 3-channel
        // input at this point and `patch_embed.norm`'s weights are `c` wide, so
        // normalising the input would read 144 channels out of a 3-channel buffer -
        // in bounds of the allocation, hence no fault, and every activation after it
        // wrong while looking plausible.
        self.cuda.channel_layer_norm(&a.body, self.d("patch_embed.norm.weight"),
                                     self.d("patch_embed.norm.bias"), &a.n1, c, hw, EPS)?;
        self.cuda.copy(&a.n1, &a.plane, c * hw)?;
        for stage in 0..self.wt.depths.len() {
            self.rhag(a, stage)?;
        }
        self.cuda.channel_layer_norm(&a.plane, self.d("norm.weight"), self.d("norm.bias"),
                                     &a.n1, c, hw, EPS)?;
        // `patch_unembed` (a reshape) then `conv_after_body`, and the conv_first
        // residual on top of it.
        self.cuda.conv3x3(&a.n1, &a.plane2, self.d("conv_after_body.weight"),
                          self.d("conv_after_body.bias"), c, c, h, w, ACT_NONE, 0.0)?;
        // The residual lands back in `plane`, which the head reads - `h0` is only
        // `feat` = 64 wide and could not hold a `c`-channel plane.
        self.cuda.add_scaled(&a.plane2, &a.body, &a.plane, 1.0, c * hw)?;
        self.head(a)
    }
}

impl Backend for Gpu<'_> {
    fn name(&self) -> &'static str {
        "gpu"
    }

    /// Mean-subtract and pad, the network, then crop and denormalise - the same entry
    /// point as the CPU backend's, so `--verify`, `--tile` and `--mem` drive either
    /// backend.
    fn forward(&mut self, h: usize, w: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        let pre = Pre::new(self.wt, h, w)?;
        let plan = pre.plan.clone();
        let adjust = pre.adjust(self.wt, input);
        let stale = match &self.acts {
            Some((p, _)) => *p != plan,
            None => true,
        };
        if stale {
            // The tile's footprint, not the image's: a tiled run calls this once per
            // tile, so `--mem` and the guard are checking the same quantity.
            crate::memguard::check_gpu(self.wt, h, w)?;
            let acts = Acts::new(self.wt, &plan)?;
            // THE PREDICTION IS HELD TO THE ALLOCATION. `footprint_floats` is what
            // the guard refused on and `footprint` is what the buffers actually
            // hold; both come from `Sizes`, so a disagreement means the buffer set
            // and the size list have drifted apart - the one failure mode that would
            // make the guard quietly wrong, and the reason the two are summed by the
            // same `total()` in the first place. A silent over-allocation is smaller
            // than the rounding here; a missing buffer is not.
            let derived = crate::gpu::footprint_floats(self.wt, &plan);
            let actual = acts.footprint();
            if actual != derived {
                return Err(format!(
                    "gpu: the buffer set does not match the guard's prediction at \
                     {}x{}: allocated {actual} floats, predicted {derived}",
                    h, w));
            }
            self.acts = Some((plan.clone(), acts));
            let a = &self.acts.as_ref().unwrap().1;
            self.maps(a)?;
        }
        {
            let a = &self.acts.as_ref().unwrap().1;
            // The input plane is re-uploaded every call, because the acts survive
            // across calls (a tiled run would otherwise reallocate the whole set per
            // tile).
            a.plane.upload(&adjust)?;
            self.forward_planes(a)?;
        }
        let a = &self.acts.as_ref().unwrap().1;
        let mut host = vec![0.0f32; a.out.bytes / 4];
        a.out.download(&mut host)?;
        Ok(finish(self.wt, &plan, &host))
    }
}
