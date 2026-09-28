//! `--cuda-selftest`: every PROJECT kernel held against a host implementation of the
//! same operator, on seeded inputs.
//!
//! WHY THIS EXISTS. `--verify` compares a backend's whole image with the reference's,
//! which catches a wrong *graph* but is a weak instrument for a wrong *kernel*: the
//! project kernels in `cuda/hat.cu` take a long argument list of shapes, and the
//! argument order is the one thing the Rust compiler cannot check. A
//! mis-ordered or mis-scaled argument produces a plausible image. Every kernel here is
//! therefore run against a host twin - the same arithmetic written twice, once in CUDA
//! and once in Rust - and the two outputs are compared elementwise.
//!
//! WHAT IS CHECKED, and how strictly:
//!
//! * `hat_window_index` vs `plan::window_index`, `hat_mask_build` vs
//!   `plan::mask_label`, `hat_oca_label`, `hat_bias_gather`, `hat_plane_edges` -
//!   INDEX AND DATA MOVEMENT, so these are compared with `==`, bit for bit and
//!   integer for integer. These are the ones a silent mis-index breaks.
//! * `hat_unfold_kv` - the same, since its only arithmetic is a select against zero.
//! * `hat_attention_d24` / `hat_attention_d30` - the only kernel with real
//!   floating-point work (`expf`, an unnormalised accumulator, one division at the
//!   end). Its host twin reproduces that order, so the comparison is against fp32
//!   rounding rather than against a re-implementation's own summation order, and it
//!   is checked at several geometries (square and NOT square, with and without a
//!   bias, masked and unmasked) because the mask and the bias are passed as integers
//!   and are the easiest arguments to shift by one.
//!
//! The kernels that MOVED INTO THE TOOLKIT are checked there instead: the toolkit's
//! own `gpuinfo` selftest holds each twin against an independent reference, and this
//! file's job is the kernels that are HAT's architecture rather than an op.
//!
//! The geometries are not random: a NON-SQUARE window grid (2 rows by 3 columns) is
//! included in the index checks because an earlier version of `Plan::nw` built maps
//! for a square grid only and every fixture in the repository was square, so nothing
//! caught it.
use lightgpu::vm::{self, DevBuf};

use crate::cuda::{upload_i32, Cuda};

/// Members of the family that have a host twin, for the failure report.
const CHECKED: [&str; 8] = [
    "hat_window_index", "hat_mask_build", "hat_oca_label", "hat_bias_gather",
    "hat_unfold_kv", "hat_plane_edges", "hat_attention_d24", "hat_attention_d30",
];

/// The tolerance for the attention kernels, and ONLY for them: their host twin
/// reproduces the kernel's own order (`expf`, accumulate `e*v`, divide once at the
/// end), so what is left is fma contraction and `expf`'s own accuracy. Everything else
/// in this module is compared exactly.
const ATTN_TOL: f32 = 1e-5;

/// One geometry to check the index kernels at. `h` and `w` need not be window
/// multiples for the LABEL kernels (they only partition a plane), but the window
/// maps are only meaningful on a window multiple, which is what the engine ever
/// builds them for.
struct Geom {
    h: usize,
    w: usize,
    win: usize,
    /// The shift for `hat_window_index` ONLY: the engine calls it at shift 0 for the
    /// even blocks as well as at `win/2` for the odd ones. `hat_mask_build` always
    /// runs at the plan's own shift (see `check_mask`).
    shift: usize,
}

fn dot3_check(name: &str, got: &[f32], want: &[f32], tol: f32) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("{name}: got {} values, expected {}", got.len(), want.len()));
    }
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if !d.is_finite() {
            // A NaN would compare false against every threshold, so it is reported as
            // the worst case explicitly rather than slipping past.
            return Err(format!("{name}: non-finite value {g} at index {i}"));
        }
        if d > worst {
            worst = d;
            at = i;
        }
    }
    if worst > tol {
        return Err(format!(
            "{name}: worst |diff| {worst:.3e} at index {at} (got {:.9e}, expected {:.9e}), \
             tolerance {tol:.0e}", got[at], want[at]));
    }
    Ok(())
}

fn int_check(name: &str, got: &[i32], want: &[i32]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("{name}: got {} values, expected {}", got.len(), want.len()));
    }
    let mut bad = 0usize;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g != w {
            bad += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    match first {
        None => Ok(()),
        Some((i, g, w)) => Err(format!(
            "{name}: {bad} of {} values differ; first at index {i}: got {g}, want {w}",
            got.len())),
    }
}

/// Download an i32 buffer. lightgpu's `DevBuf` has no i32 download, so the bytes come
/// back as f32 and are bit-cast - which is exact, and is why the kernels can be
/// compared as integers rather than as floats with a tolerance.
fn download_i32(b: &DevBuf, n: usize) -> Result<Vec<i32>, String> {
    let mut f = vec![0.0f32; n];
    b.download(&mut f)?;
    Ok(f.iter().map(|v| v.to_bits() as i32).collect())
}

/// The device's `hat_window_index` against `plan::window_index` at one geometry.
fn check_window_index(cu: &Cuda, g: &Geom) -> Result<(), String> {
    let plan = crate::plan::Plan::new(g.h, g.w, g.win, 64);
    let nw = plan.nw();
    let win = plan.win;
    let total = nw * win * win;
    let out = DevBuf::alloc(total * 2 * 4)?;
    cu.window_index(&out, nw, plan.nx(), win, g.h, g.w, g.shift)?;
    let got = download_i32(&out, total * 2)?;
    let want = crate::plan::window_index(&plan, g.shift);
    // The host map is a LINEAR plane index (`src_y * wp + src_x`); the kernel writes
    // the (y, x) pair, so the comparison splits it here rather than converting the
    // device's output (which is the thing being checked).
    let mut want_i = Vec::with_capacity(want.len() * 2);
    for &p in &want {
        want_i.push((p / plan.wp as u32) as i32);
        want_i.push((p % plan.wp as u32) as i32);
    }
    int_check(&format!("hat_window_index {}x{} shift {}", g.h, g.w, g.shift), &got, &want_i)
}

/// The device's window/mask labels against `plan::mask_label`.
///
/// THE SHIFT HERE IS `plan.shift`, NOT THE GEOMETRY'S. `Plan::new` derives
/// `shift = win/2`, and `plan::mask_label` labels the plane with THAT shift, while
/// `Geom::shift` exists to exercise `hat_window_index` - which is called at shift 0
/// for the even blocks as well. Passing the geometry's shift (0) to `hat_mask_build`
/// while comparing against the host's shift-8 labels is a real mismatch and not a
/// kernel bug: at shift 0 the last two bands coincide, so the kernel correctly yields
/// four regions where the host yields nine. `gpu.rs::maps` never builds a mask at all
/// when `plan.shift == 0`, and this mirrors that.
fn check_mask(cu: &Cuda, g: &Geom) -> Result<(), String> {
    let plan = crate::plan::Plan::new(g.h, g.w, g.win, 64);
    let nw = plan.nw();
    let total = nw * plan.win * plan.win;
    let labels = DevBuf::alloc(total * 4)?;
    cu.mask_build(&labels, nw, plan.nx(), plan.win, g.h, g.w, plan.shift)?;
    let got = download_i32(&labels, total)?;
    let want: Vec<i32> = crate::plan::mask_label(&plan)
        .iter()
        .map(|v| *v as i32)
        .collect();
    int_check(&format!("hat_mask_build {}x{} shift {}", g.h, g.w, plan.shift), &got, &want)
}

/// `hat_bias_gather` against the reference's own expression, written out.
fn check_bias_gather(cu: &Cuda) -> Result<(), String> {
    let (nq, nk, heads, rows) = (5usize, 7usize, 3usize, 4usize);
    let tab: Vec<f32> = (0..rows * heads).map(|i| i as f32 * 0.25 - 1.0).collect();
    // THE INDICES ARE UPLOADED ALREADY FOLDED, exactly as the graph does it:
    // `plan::rpi_sa`/`rpi_oca` produce values that PyTorch indexes negatively, and
    // the engine folds them with `rem_euclid` into the table's row count BEFORE the
    // upload (see `wrap_i64` in `gpu.rs`). The KERNEL DOES NOT FOLD - it computes
    // `tab[(long)p * heads + hd]` on whatever it is given - so uploading a raw -1
    // here would have the kernel read gigabytes past the table and return whatever
    // the allocator held (0.0, in practice). That is a bug in a CHECK, not in the
    // kernel, and it is worth knowing which of the two produced a disagreement.
    let raw: Vec<i64> = (0..nq * nk).map(|i| ((i * 7) % rows) as i64 - 1).collect();
    let idx: Vec<i32> = raw.iter().map(|v| v.rem_euclid(rows as i64) as i32).collect();
    let (dt, di, dout) = (
        DevBuf::from_host(&tab)?,
        upload_i32(&idx)?,
        DevBuf::alloc(heads * nq * nk * 4)?,
    );
    cu.bias_gather(&dt, &di, &dout, nq, nk, heads)?;
    let got = {
        let mut v = vec![0.0f32; heads * nq * nk];
        dout.download(&mut v)?;
        v
    };
    // `out[(hd*nk + k)*nq + q] = tab[p*heads + hd]` - the TRANSPOSED layout the
    // attention reads, so the host side writes it transposed too.
    let mut want = vec![0.0f32; heads * nq * nk];
    for hd in 0..heads {
        for q in 0..nq {
            for k in 0..nk {
                // `rem_euclid` here reproduces the fold the engine applied before the
                // upload; on already-folded values it is the identity, so this is a
                // restatement of the same convention rather than a second one.
                let p = idx[q * nk + k].rem_euclid(rows as i32) as usize;
                want[(hd * nk + k) * nq + q] = tab[p * heads + hd];
            }
        }
    }
    dot3_check("hat_bias_gather", &got, &want, 0.0)
}

/// `hat_unfold_kv` against `nn.Unfold`'s own indexing.
fn check_unfold(cu: &Cuda, h: usize, w: usize, win: usize, owin: usize) -> Result<(), String> {
    let plan = crate::plan::Plan::new(h, w, win, 64);
    let nw = plan.nw();
    let opad = plan.opad();
    let c = 3usize;
    let wk = owin * owin;
    let plane: Vec<f32> = (0..2 * c * h * w).map(|i| ((i * 13) % 97) as f32 * 0.5 - 20.0).collect();
    let (dp, dk, dv) = (
        DevBuf::from_host(&plane)?,
        DevBuf::alloc(nw * wk * c * 4)?,
        DevBuf::alloc(nw * wk * c * 4)?,
    );
    cu.unfold_kv(&dp, &dk, &dv, nw, plan.nx(), win, owin, opad, h, w, c)?;
    let mut got_k = vec![0.0f32; nw * wk * c];
    let mut got_v = vec![0.0f32; nw * wk * c];
    dk.download(&mut got_k)?;
    dv.download(&mut got_v)?;
    let mut want_k = vec![0.0f32; nw * wk * c];
    let mut want_v = vec![0.0f32; nw * wk * c];
    for wl in 0..nw {
        let (wh, ww) = (wl / plan.nx(), wl % plan.nx());
        for t in 0..wk {
            let (i, j) = (t / owin, t % owin);
            let y = wh * win + i - opad;
            let x = ww * win + j - opad;
            for ch in 0..c {
                let val = if y < h && x < w {
                    plane[ch * h * w + y * w + x]
                } else {
                    0.0
                };
                want_k[(wl * wk + t) * c + ch] = val;
                let val_v = if y < h && x < w {
                    plane[(c + ch) * h * w + y * w + x]
                } else {
                    0.0
                };
                want_v[(wl * wk + t) * c + ch] = val_v;
            }
        }
    }
    dot3_check(&format!("hat_unfold_kv k {h}x{w}"), &got_k, &want_k, 0.0)?;
    dot3_check(&format!("hat_unfold_kv v {h}x{w}"), &got_v, &want_v, 0.0)
}

/// `hat_plane_edges` in place: the `pad`-wide border of every channel goes to zero
/// and nothing else moves.
fn check_edges(cu: &Cuda) -> Result<(), String> {
    let (c, hp, wp, pad) = (2usize, 6usize, 7usize, 2usize);
    let src: Vec<f32> = (0..c * hp * wp).map(|i| i as f32 + 1.0).collect();
    let d = DevBuf::from_host(&src)?;
    cu.plane_edges(&d, c, hp, wp, pad)?;
    let mut got = vec![0.0f32; c * hp * wp];
    d.download(&mut got)?;
    let mut want = src.clone();
    for ch in 0..c {
        for y in 0..hp {
            for x in 0..wp {
                if y < pad || y >= hp - pad || x < pad || x >= wp - pad {
                    want[(ch * hp + y) * wp + x] = 0.0;
                }
            }
        }
    }
    dot3_check("hat_plane_edges", &got, &want, 0.0)
}

/// `hat_oca_label` against its definition.
fn check_oca_label(cu: &Cuda, h: usize, w: usize, win: usize, owin: usize) -> Result<(), String> {
    let plan = crate::plan::Plan::new(h, w, win, 64);
    let nw = plan.nw();
    let opad = plan.opad();
    let wk = owin * owin;
    let labels = DevBuf::alloc(nw * wk * 4)?;
    cu.oca_label(&labels, nw, plan.nx(), win, owin, opad, h, w)?;
    let got = download_i32(&labels, nw * wk)?;
    let mut want = vec![0i32; nw * wk];
    for wl in 0..nw {
        let (wh, ww) = (wl / plan.nx(), wl % plan.nx());
        for t in 0..wk {
            let (i, j) = (t / owin, t % owin);
            let y = wh * win + i;
            let x = ww * win + j;
            let inside = (y as isize - opad as isize) >= 0
                && (y as isize - opad as isize) < h as isize
                && (x as isize - opad as isize) >= 0
                && (x as isize - opad as isize) < w as isize;
            want[wl * wk + t] = if inside { 0 } else { 9 };
        }
    }
    int_check(&format!("hat_oca_label {h}x{w}"), &got, &want)
}

/// The attention, against a host twin that reproduces the kernel's own order.
#[allow(clippy::too_many_arguments)]
fn check_attention(
    cu: &Cuda, heads: usize, d: usize, nw: usize, nq: usize, nk: usize,
    has_bias: bool, masked: bool,
) -> Result<(), String> {
    let c = heads * d;
    let wave = |n: usize, s: usize| -> Vec<f32> {
        (0..n).map(|i| ((((i * s) % 211) as f32) / 211.0 - 0.5) * 1.5).collect()
    };
    let q = wave(nw * nq * c, 3);
    let k = wave(nw * nk * c, 5);
    let v = wave(nw * nk * c, 7);
    let bias = if has_bias { wave(heads * nk * nq, 11) } else { Vec::new() };
    // A label grid with several distinct regions, so the mask actually fires.
    let labels: Vec<i32> = (0..nw * nq)
        .map(|i| ((i / 3) % 4) as i32)
        .collect();
    let scale = 1.0f32 / (d as f32).sqrt();

    let (dq, dk, dv) = (
        DevBuf::from_host(&q)?,
        DevBuf::from_host(&k)?,
        DevBuf::from_host(&v)?,
    );
    let db = if has_bias { Some(DevBuf::from_host(&bias)?) } else { None };
    let dl = upload_i32(&labels)?;
    let dout = DevBuf::alloc(nw * nq * c * 4)?;
    let name = format!("hat_attention_d{d}");
    // `Cuda::attention` takes NO kernel name: it resolves d24/d30 from `d` itself, so
    // an unknown head dim is a launch error rather than a silent wrong-size fallback.
    // `nww` precedes `nq` even though the kernel ignores it.
    cu.attention(&dq, &dk, &dv, db.as_ref(), if masked { Some(&dl) } else { None },
                 &dout, nw, 1, nq, nk, heads, d, scale, has_bias, masked)?;
    let mut got = vec![0.0f32; nw * nq * c];
    dout.download(&mut got)?;

    let mut want = vec![0.0f32; nw * nq * c];
    for wl in 0..nw {
        let lab = &labels[wl * nq..(wl + 1) * nq];
        for qi in 0..nq {
            for hd in 0..heads {
                let qr = &q[(wl * nq + qi) * c + hd * d..(wl * nq + qi) * c + hd * d + d];
                let mut max = f32::NEG_INFINITY;
                let mut srow = vec![0.0f32; nk];
                for ki in 0..nk {
                    let kr = &k[(wl * nk + ki) * c + hd * d..(wl * nk + ki) * c + hd * d + d];
                    let mut s = 0.0f32;
                    for i in 0..d {
                        s += qr[i] * kr[i];
                    }
                    s = s * scale
                        + if has_bias { bias[(hd * nk + ki) * nq + qi] } else { 0.0 };
                    if masked && lab[ki] != lab[qi] {
                        s -= 100.0;
                    }
                    srow[ki] = s;
                    if s > max {
                        max = s;
                    }
                }
                let mut sum = 0.0f32;
                let mut acc = vec![0.0f32; d];
                for ki in 0..nk {
                    let e = (srow[ki] - max).exp();
                    sum += e;
                    let vr = &v[(wl * nk + ki) * c + hd * d..(wl * nk + ki) * c + hd * d + d];
                    for i in 0..d {
                        acc[i] += e * vr[i];
                    }
                }
                let inv = 1.0 / sum;
                for i in 0..d {
                    want[(wl * nq + qi) * c + hd * d + i] = acc[i] * inv;
                }
            }
        }
    }
    dot3_check(
        &format!("{name} heads {heads} d {d} nw {nw} nq {nq} nk {nk} \
                  bias {has_bias} mask {masked}"),
        &got, &want, ATTN_TOL)
}

/// `lg_conv3x3_winograd` against `lg_conv3x3s1p1`, the toolkit's own direct kernel.
///
/// This is the one check here that is NOT a project kernel, and it is not redundant:
/// the winograd launch passes `ocb` and `c_chunk` as RUNTIME arguments that the kernel
/// trusts, so a caller that disagrees with it produces a wrong image with no fault and
/// no diagnostic (the `shared` size is computed from the same two numbers, so a
/// mismatch in one place is a mismatch in both). Comparing against the direct form is
/// what makes those two constants observable.
///
/// THE TOLERANCE IS NOT ZERO AND MUST NOT BE: F(4,3) winograd transforms the input,
/// the weights and the output, so it is a different arithmetic route to the same
/// convolution and its rounding differs from the direct sum's. `3e-4` absolute at
/// these magnitudes is well inside what `--verify`'s end-to-end `1e-4` requires of the
/// whole graph - the point is to catch a TRANSFORM or INDEXING error (which is 1e-1),
/// not the last bit.
fn check_winograd(cu: &Cuda, ci: usize, co: usize, h: usize, w: usize) -> Result<f32, String> {
    let hw = h * w;
    let wave = |n: usize, s: usize| -> Vec<f32> {
        (0..n).map(|i| (((i * s) % 211) as f32) / 211.0 - 0.5).collect()
    };
    let inp = wave(ci * hw, 3);
    let wts = wave(co * ci * 9, 5);
    let bia = wave(co, 7);
    let (di, dw, db) = (
        DevBuf::from_host(&inp)?,
        DevBuf::from_host(&wts)?,
        DevBuf::from_host(&bia)?,
    );
    let dwg = DevBuf::alloc(co * hw * 4)?;
    let ddir = DevBuf::alloc(co * hw * 4)?;
    cu.conv3x3(&di, &dwg, &dw, &db, ci, co, h, w, 0, 0.0)?;
    cu.conv3x3_direct(&di, &ddir, &dw, &db, ci, co, h, w)?;
    let mut got = vec![0.0f32; co * hw];
    let mut want = vec![0.0f32; co * hw];
    dwg.download(&mut got)?;
    ddir.download(&mut want)?;
    // `c_chunk = 4` and `ocb = 16` are the constants `Cuda::conv3x3` uses; the
    // reduction length `ci*9` cancels in the relative error, so an absolute
    // comparison at unit-magnitude inputs is the right shape here.
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    dot3_check(&format!("lg_conv3x3_winograd vs direct {ci}->{co} {h}x{w}"), &got, &want, 3e-4)?;
    Ok(worst)
}

/// Run every check and print a line per pass. `Err` on the first failure, so the
/// message names the operator rather than a whole-image difference.
pub fn run() -> Result<(), String> {
    let cu = Cuda::new()?;
    let missing: Vec<&str> = CHECKED
        .iter()
        .filter(|n| cu.module_of(n).is_err())
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "this build's fatbin has no {}: build.rs's PROJECT_KERNELS and the kernels in \
             cuda/hat.cu disagree",
            missing.join(", ")));
    }
    let mut passed = 0usize;
    let mut note = |name: &str| {
        println!("  ok  {name}");
        passed += 1;
    };

    // Square and NON-SQUARE, and a shift that is not half the window, so the modulo
    // wrap and the per-axis window counts are both exercised.
    for g in [
        Geom { h: 32, w: 32, win: 16, shift: 0 },
        Geom { h: 32, w: 32, win: 16, shift: 8 },
        Geom { h: 32, w: 48, win: 16, shift: 8 },
        Geom { h: 48, w: 32, win: 16, shift: 0 },
        Geom { h: 48, w: 48, win: 16, shift: 8 },
    ] {
        check_window_index(&cu, &g)?;
        note(&format!("hat_window_index  {}x{} shift {}", g.h, g.w, g.shift));
        check_mask(&cu, &g)?;
        note(&format!("hat_mask_build    {}x{} shift {}", g.h, g.w, crate::plan::Plan::new(g.h, g.w, g.win, 64).shift));
        if g.shift == 8 {
            check_oca_label(&cu, g.h, g.w, g.win, 24)?;
            note(&format!("hat_oca_label     {}x{}", g.h, g.w));
            check_unfold(&cu, g.h, g.w, g.win, 24)?;
            note(&format!("hat_unfold_kv     {}x{}", g.h, g.w));
        }
    }
    // The winograd launch's own contract, including a `co` that is NOT a multiple of
    // `ocb` (150 = 9*16 + 6, so the last output block is partly out of range - the
    // case a hardcoded `ceil` gets wrong) and the CAB's small shapes.
    for (ci, co) in [(144usize, 144usize), (144, 150), (144, 6), (64, 3), (140, 144)] {
        let worst = check_winograd(&cu, ci, co, 32, 32)?;
        // The margin is printed because it is the evidence that the tolerance is
        // generous rather than tight: winograd's transform error is at fp32 epsilon
        // scale, while a wrong `ocb`/`c_chunk` or a wrong transform is ~1e-1.
        note(&format!("lg_conv3x3_winograd {ci}->{co} 32x32   worst {worst:.2e} vs tol 3e-4"));
    }

    check_bias_gather(&cu)?;
    note("hat_bias_gather");
    check_edges(&cu)?;
    note("hat_plane_edges");

    // The attention at both head dims the family uses, square and non-square, with
    // and without a bias and a mask.
    //
    // A MASK IS ONLY CHECKED WHERE nk == nq, which is not an oversight: the mask is
    // indexed by the QUERY's window token (`labels[wl*nq + ki]`), so a key outside
    // that window's `win*win` labels has no label to compare against. The graph never
    // asks for it - the masked blocks are the HABs, whose keys and queries are both
    // the same `win*win` window (256), while the OCAB's 576 keys against 256 queries
    // are never masked - so the combinations exercised here are the ones the engine
    // actually uses rather than a synthetic cross-product.
    for d in [24usize, 30] {
        for (nw, nq, nk) in [
            (2usize, 16usize, 16usize),   // square windows, the masked HAB's shape
            (3, 16, 16),                  // a non-square window grid
            (1, 25, 49),                  // the OCAB's shape: MORE KEYS THAN QUERIES
            (2, 4, 4),                    // a window smaller than a warp
            (2, 4, 9),                    // ... and its overlapping twin
        ] {
            for (hb, mk) in [(false, false), (true, false), (true, true)] {
                if mk && nk != nq {
                    continue;
                }
                check_attention(&cu, 6, d, nw, nq, nk, hb, mk)?;
                note(&format!("hat_attention_d{d}   nw {nw} nq {nq} nk {nk} \
                               bias {hb} mask {mk}"));
            }
        }
    }
    println!("  cuda selftest: {passed} checks passed");
    let _ = vm::free_vram();
    Ok(())
}
