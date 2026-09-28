// HAT's own kernels: the parts of this architecture that the lightgpu toolkit
// does not (and should not) provide.
//
// The toolkit has the generic op set - 3x3 and 1x1 convolutions, matmuls, layer
// norms, activations, the window gather/scatter pair, and the depth-to-space the
// reconstruction head's octaves need - and this file has the four things that are
// HAT's architecture rather than an op:
//
//   * `hat_attention`, the windowed attention with a RELATIVE-POSITION BIAS and,
//     for the odd blocks, the shifted-window MASK. The toolkit's `lg_attn_*` are
//     causal/grouped LLM attentions; this one has `q` from one window grid and,
//     in the overlapping cross attention, `k`/`v` from a LARGER strided grid.
//   * `hat_bias_gather`, which materialises `relative_position_bias_table[rpi]`
//     as `[heads][nq][nk]`. The reference indexes the table with the checkpoint's
//     own index buffer, so the gather is part of the model.
//   * `hat_mask_build`, the shifted-window mask as the LABEL GRID it is a
//     function of - `[nw][win*win]` u32 rather than the `[nw][win][win]` float
//     addend the reference materialises, which at 1024x1024 is 1.07 GB.
//   * `hat_unfold_kv`, the overlapping cross attention's key/value windows:
//     `nn.Unfold(owin, stride=win, padding=(owin-win)/2)` over the projected
//     key/value plane, i.e. a 13x13 (24x24 kernel) neighbourhood of a
//     16-strided grid, which is NOT the toolkit's window gather.
//
// `hat_window_index` is here for the selftest rather than for the graph: it
// writes the device's own window coordinate map so `--cuda-selftest` can hold it
// against `plan::window_index`, the host map the CPU backend indexes with. The
// gather the graph uses is the toolkit's, which folds the same map in.
//
// EVERY KERNEL HERE IS CHECKED AGAINST A HOST TWIN by `--cuda-selftest`: a kernel
// that merely produces a plausible image would otherwise pass `--verify` on a
// lucky day.
//
// WHAT USED TO BE HERE AND IS NOT. Three kernels were restatements of toolkit ops
// rather than architecture, and were measured against them before removal: the
// pixel shuffle of the reconstruction head is now `lg_pixel_shuffle`, and
// `hat_plane_block`/`hat_token_range`/`hat_plane_bias` are `lg_channel_affine`
// (with the channel range as a pointer offset) and `lg_extract_rows`. Each A/B was
// interleaved in one process and read a tie - 1.000-1.007 - and the shims that keep
// the graph's call sites readable are in `src/cuda.rs` with the numbers.

#include <math.h>

// ---------------------------------------------------------------------------
// Window geometry
// ---------------------------------------------------------------------------

// The source coordinate of window token `t` of window `wl`, at a cyclic shift.
// This is `lg_window_index`'s map, repeated here so the selftest can compare the
// device's arithmetic with `plan::window_index` without depending on the
// toolkit's private copy. `nww` is the number of window COLUMNS, so a
// non-square plane (32x48 is 2 window rows by 3 columns) divides correctly.
__device__ void hat_win_coord(int wl, int t, int nww, int win, int hp, int wp, int shift,
                              int *py, int *px)
{
    const int i = t / win;
    const int j = t % win;
    const int wh = wl / nww;
    const int ww = wl % nww;
    *py = (wh * win + i + shift) % hp;
    *px = (ww * win + j + shift) % wp;
}

// out[2*(wl*win*win + t) + 0] = y, + 1 = x. Grid over nw*win*win.
extern "C" __global__ void hat_window_index(
    int *__restrict__ out, int nw, int nww, int win, int hp, int wp, int shift)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)nw * win * win;
    if (idx >= total) return;
    const int t = (int)(idx % (win * win));
    const int wl = (int)(idx / (win * win));
    int y, x;
    hat_win_coord(wl, t, nww, win, hp, wp, shift, &y, &x);
    out[2 * idx] = y;
    out[2 * idx + 1] = x;
}

// ---------------------------------------------------------------------------
// The relative-position bias
// ---------------------------------------------------------------------------

// out[hd][q][k] = tab[idx[q*nk + k] * heads + hd], i.e. the reference's
// `relative_position_bias_table[rpi.view(-1)].view(nq, nk, nH).permute(2, 0, 1)`.
// The map is the checkpoint's own index buffer (folded to u32 where it is
// negative, which selects the row PyTorch's negative indexing would), so the
// gather is the model's own definition rather than a convention of ours.
//
// Grid over heads*nq*nk, one element per thread: every element of the output is
// one indirection into a table that a whole head shares, so the table stays hot
// in L2 across the block. 3.5 MB per overlapping-attention call at the sizes
// this runs, which is why the graph gathers once per call rather than once per
// score.
// The gathered table is TRANSPOSED - `[heads][nk][nq]`, not `[heads][nq][nk]` -
// because that is the order the attention reads it in: its warp's lanes are
// consecutive QUERIES, so `bias[(hd*nk + ki)*nq + qi]` is 32 consecutive floats and
// one 128-byte transaction per key, where the natural order would stride by `nk`
// and give every lane its own sector.
extern "C" __global__ void hat_bias_gather(
    const float *__restrict__ tab, const int *__restrict__ idx, float *__restrict__ out,
    int nq, int nk, int heads)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)heads * nq * nk;
    if (i >= total) return;
    const int k = (int)(i % nk);
    const long t2 = i / nk;
    const int q = (int)(t2 % nq);
    const int hd = (int)(t2 / nq);
    const int p = idx[q * nk + k];
    // Written to `(hd, k, q)` while read from `(hd, q, k)` above.
    out[((size_t)hd * nk + k) * nq + q] = tab[(long)p * heads + hd];
}

// ---------------------------------------------------------------------------
// The shifted-window mask
// ---------------------------------------------------------------------------

// The reference's `calculate_mask`, up to the comparison: the label of every
// window token. `labels[wl*win*win + t]` is `ry*3 + rx` of the nine rectangles
// the three slice pairs cut the plane into, exactly as the reference labels it.
//
// THE LABELS RATHER THAN THE MASK, because the mask is `nw*win^4` floats and the
// labels are `nw*win^2` u32: 4 MB against 1.07 GB at 1024x1024. The attention
// derives the addend with one integer compare per score, which is what makes the
// engine's largest allocation proportional to `nw*win^2`.
//
// Grid over nw*win*win.
// THE SHIFT ARGUMENT GOVERNS THE BANDS, NOT THE COORDINATES, and getting that
// backwards is silent. The reference's `calculate_mask` labels the plane with
// `h_slices = (slice(0, -win), slice(-win, -shift), slice(-shift, None))` - so the
// three BANDS depend on the shift - and then partitions that label grid at each
// window's UNROLLED position, because the roll is applied to `x` and never to the
// mask. A single `shift` used for both would either roll the coordinates (masking
// the wrong token pairs) or collapse the bands to two, which is what the
// `shift = 0` call this replaced did: the label dump had four distinct values where
// the shift-dependent grid needs nine.
extern "C" __global__ void hat_mask_build(
    int *__restrict__ labels, int nw, int nww, int win, int hp, int wp, int shift)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)nw * win * win;
    if (idx >= total) return;
    const int t = (int)(idx % (win * win));
    const int wl = (int)(idx / (win * win));
    int y, x;
    hat_win_coord(wl, t, nww, win, hp, wp, 0, &y, &x);
    // The reference's `h_slices = (slice(0, -win), slice(-win, -shift), slice(-shift, None))`
    // on each axis: which of the three bands the coordinate falls in.
    const int ry = y < hp - win ? 0 : (y < hp - shift ? 1 : 2);
    const int rx = x < wp - win ? 0 : (x < wp - shift ? 1 : 2);
    labels[idx] = ry * 3 + rx;
}

// ---------------------------------------------------------------------------
// The overlapping cross attention's keys and values
// ---------------------------------------------------------------------------

// The projected key/value plane `[2c][hp][wp]` unfolded into windows of
// `owin x owin` at stride `win`, with `pad = (owin - win) / 2` and zeros outside
// the plane - `nn.Unfold(kernel_size=owin, stride=win, padding=pad)`, which is
// the reference's `unfold` on the `cat(k, v)` plane, in its C-major order.
//
// `plane` holds k in channels [0, c) and v in [c, 2c), and the two are unpacked
// here into separate window buffers because the attention reads them separately.
// A position outside the plane contributes a zero AND is not read from the
// plane: the reference's zero padding is real padding, and its gathered
// neighbours are what the OCAB attends over.
//
// Grid over nw*wk*2c, one element per thread.
extern "C" __global__ void hat_unfold_kv(
    const float *__restrict__ plane, float *__restrict__ kw, float *__restrict__ vw,
    int nw, int nww, int win, int owin, int pad, int hp, int wp, int c)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int cc = 2 * c;
    const long wk = (long)owin * owin;
    const long total = (long)nw * wk * cc;
    if (idx >= total) return;
    const int ch = (int)(idx % cc);
    const long t2 = idx / cc;
    const int t = (int)(t2 % wk);
    const int wl = (int)(t2 / wk);
    const int wh = wl / nww;
    const int ww = wl % nww;
    const int i = t / owin;
    const int j = t % owin;
    const int y = wh * win - pad + i;
    const int x = ww * win - pad + j;
    float val = 0.0f;
    if (y >= 0 && y < hp && x >= 0 && x < wp) {
        val = plane[(size_t)ch * hp * wp + (size_t)y * wp + x];
    }
    if (ch < c) {
        kw[(size_t)wl * wk * c + (size_t)t * c + ch] = val;
    } else {
        vw[(size_t)wl * wk * c + (size_t)t * c + (ch - c)] = val;
    }
}

// ---------------------------------------------------------------------------
// The windowed attention
// ---------------------------------------------------------------------------

// ONE BLOCK PER (window, head), ONE THREAD PER QUERY, AND THE HEAD IS THE SLOW
// INDEX. What it replaces was one thread per (window, query, head), and cuobjdump
// showed what that cost on the 1080:
//
//   * `float acc[64]` with `d` a RUNTIME argument made the compiler reserve the
//     worst case - the kernel reported REG:40 STACK:256, and 256 bytes is exactly
//     `acc[64]`. THE ACCUMULATOR WAS IN LOCAL MEMORY, so every `acc[i] += e*vr[i]`
//     was a local load-modify-store, and the stack also capped how many blocks fit
//     per SM. Templating on `d` (24 for hat-s, 30 for hat and hat-l - the only
//     values in the family) sizes `acc` at compile time, so it lives in registers
//     and both loops unroll fully.
//   * `hd = idx % heads` made the HEAD the fast index, so the 32 lanes of a warp
//     spanned all 6 head offsets and every k/v load was a 6-way gather instead of a
//     broadcast. With one head per block, `qr`/`kr`/`vr` differ only by `qi`, which
//     is not used in their address at all - a warp's key and value loads are one
//     address broadcast to 32 lanes.
//
// THE BIAS IS TRANSPOSED. It is gathered as `[heads][nk][nq]` rather than
// `[heads][nq][nk]` precisely so that a warp - whose lanes are consecutive queries
// - reads `bias[(hd*nk + ki)*nq + qi]`, 32 consecutive floats, one 128-byte
// transaction. In the natural order the same read strides by `nk` and every lane
// touches its own sector.
//
// The arithmetic per row, in the reference's order, is UNCHANGED, so the result is
// bit for bit what the previous kernel produced:
//   s[k] = (q . k[k]) * scale + bias[hd][q][k]        [+ mask, if the block is odd]
//   p = softmax(s)                                    [max subtracted, as the reference does]
//   out[hd] = sum_k p[k] * v[k]
// `scale` is `head_dim ** -0.5`, applied to q BEFORE the matmul as the reference
// does (`q = q * self.scale`), not folded into the softmax. The row of scores is
// computed TWICE rather than stored: a row is `nk` floats - 576 in the overlapping
// form - and holding one per query would be 256 KB for a single window.
//
// THE MASK IS DERIVED FROM THE LABELS rather than read as a `[nw][nq][nk]`
// addend: `labels[wl*nq + q] != labels[wl*nq + k]` is one integer compare and
// `-100.0` is what the reference fills in. The GPU has the compare to spare and
// not the memory. Within a block only `qi` varies, so the KEY label read is a
// broadcast too.
//
// grid = (nw * heads, 1, 1), block = (nq, 1, 1). The two instantiations are the
// only `head_dim`s the family uses, and the Rust side picks by name - a build with
// a third variant would fail at launch rather than silently use the wrong one.
template <int D>
__device__ __forceinline__ void hat_attn_impl(
    const float *__restrict__ q, const float *__restrict__ kw, const float *__restrict__ vw,
    const float *__restrict__ bias, const int *__restrict__ labels, float *__restrict__ out,
    int nw, int nq, int nk, int heads, float scale, int has_bias, int masked)
{
    const int hd = blockIdx.x % heads;
    const int wl = blockIdx.x / heads;
    const int qi = threadIdx.x;
    if (qi >= nq) return;
    const int c = heads * D;
    const float *qr = q + ((size_t)wl * nq + qi) * c + hd * D;
    const float *brow = bias + (size_t)hd * nk * nq;
    const int use_mask = (masked && labels != 0) ? 1 : 0;
    const int lab_q = use_mask ? labels[wl * nq + qi] : 0;

    // Pass 1: the row's maximum, for the numerical stability the reference gets
    // from `softmax(x, dim=-1)`'s own max subtraction.
    float max = -INFINITY;
    for (int ki = 0; ki < nk; ++ki) {
        const float *kr = kw + ((size_t)wl * nk + ki) * c + hd * D;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < D; ++i) s += qr[i] * kr[i];
        s = s * scale + (has_bias ? brow[(size_t)ki * nq + qi] : 0.0f);
        if (use_mask && labels[wl * nq + ki] != lab_q) s -= 100.0f;
        max = s > max ? s : max;
    }

    // Pass 2: exponentiate, sum, and accumulate the weighted values. The
    // accumulator is divided by the sum at the end rather than normalising each
    // probability first, which is one fewer multiply per key and the same result
    // to within the fp32 rounding the parity tolerance already allows.
    float sum = 0.0f;
    float acc[D];
#pragma unroll
    for (int i = 0; i < D; ++i) acc[i] = 0.0f;
    for (int ki = 0; ki < nk; ++ki) {
        const float *kr = kw + ((size_t)wl * nk + ki) * c + hd * D;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < D; ++i) s += qr[i] * kr[i];
        s = s * scale + (has_bias ? brow[(size_t)ki * nq + qi] : 0.0f);
        if (use_mask && labels[wl * nq + ki] != lab_q) s -= 100.0f;
        const float e = expf(s - max);
        sum += e;
        const float *vr = vw + ((size_t)wl * nk + ki) * c + hd * D;
#pragma unroll
        for (int i = 0; i < D; ++i) acc[i] += e * vr[i];
    }
    const float inv = 1.0f / sum;
    float *orow = out + ((size_t)wl * nq + qi) * c + hd * D;
#pragma unroll
    for (int i = 0; i < D; ++i) orow[i] = acc[i] * inv;
}

extern "C" __global__ void hat_attention_d24(
    const float *__restrict__ q, const float *__restrict__ kw, const float *__restrict__ vw,
    const float *__restrict__ bias, const int *__restrict__ labels, float *__restrict__ out,
    int nw, int nq, int nk, int heads, float scale, int has_bias, int masked)
{
    hat_attn_impl<24>(q, kw, vw, bias, labels, out, nw, nq, nk, heads, scale, has_bias, masked);
}

extern "C" __global__ void hat_attention_d30(
    const float *__restrict__ q, const float *__restrict__ kw, const float *__restrict__ vw,
    const float *__restrict__ bias, const int *__restrict__ labels, float *__restrict__ out,
    int nw, int nq, int nk, int heads, float scale, int has_bias, int masked)
{
    hat_attn_impl<30>(q, kw, vw, bias, labels, out, nw, nq, nk, heads, scale, has_bias, masked);
}



// ---------------------------------------------------------------------------
// The zero border the overlapping attention's unfold reads
// ---------------------------------------------------------------------------

// ZERO THE `pad`-wide BORDER of every channel of a plane.
//
// WHY THIS IS NEEDED, and why it is not the same thing as padding a buffer. The
// reference's overlapping attention unfolds `nn.Unfold(kernel, stride, padding)`,
// so the 13-float step off the plane reads a ZERO. The engine's window gather folds
// its index map with a modulo (`(ww*win + j + shift) % wp`), so the same step reads
// the FAR EDGE of the plane instead - a real value, not a zero, and on a reflection-
// padded image a plausible one.
//
// The fix has to be cheap and it has to keep the plane's geometry: a genuine
// zero-padded buffer would need its own `lg_window_size + owin` shape and would make
// the index map's moduli wrong. Instead the edges are zeroed IN PLACE, once per
// block, which makes every wrapped read a zero. Only `pad` rows and columns at each
// edge - the whole range the kernel can step outside by - so a plane built for
// `padding = (owin - win) / 2` becomes exactly the padded plane the reference reads,
// at the cost of one pass over the kv plane per overlapping block.
//
// The two bands do not overlap because `pad = (owin - win) / 2 = 4 < win = 16`, so a
// zero here is never a value a real key needs.
//
// grid = (ceil(c*hp*wp / 256), 1, 1).
extern "C" __global__ void hat_plane_edges(
    float *__restrict__ plane, int c, int hp, int wp, int pad)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)c * hp * wp) return;
    const int x = (int)(i % wp);
    const int y = (int)((i / wp) % hp);
    if (y < pad || y >= hp - pad || x < pad || x >= wp - pad) plane[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// The overlapping attention's key labels
// ---------------------------------------------------------------------------

// `[nw][owin*owin]` i32: 0 for a key INSIDE the plane, 9 for a key the unfold
// reached through its zero padding. The attention masks a key whose label differs
// from the query's, so every padded key is masked for every query.
//
// WHY THIS IS A LABEL AND NOT A BIAS. A padded key has `k = 0`, so its score is
// `0 + bias`; making the bias zero there would be the other way to neutralise it -
// which is what the CPU backend does, by clamping the relative-position index at a
// padded slot to the table's zero row. A LABEL is better here because it does not
// depend on the table containing a zero, and because the attention already derives
// a mask from labels: making padded keys merely "some other region" reuses that
// machinery with no new state.
//
// Labels 0..8 are the reference's nine `calculate_mask` regions, so 9 is free.
//
// grid = (ceil(nw*owin*owin / 256), 1, 1).
extern "C" __global__ void hat_oca_label(
    int *__restrict__ labels, int nw, int nww, int win, int owin, int pad, int hp, int wp)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long wk = (long)owin * owin;
    if (idx >= (long)nw * wk) return;
    const int t = (int)(idx % wk);
    const int wl = (int)(idx / wk);
    const int wh = wl / nww;
    const int ww = wl % nww;
    const int i = t / owin;
    const int j = t % owin;
    const int y = wh * win - pad + i;
    const int x = ww * win - pad + j;
    labels[idx] = (y >= 0 && y < hp && x >= 0 && x < wp) ? 0 : 9;
}


