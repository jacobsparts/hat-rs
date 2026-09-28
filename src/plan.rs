//! The geometry of one forward pass, and the index maps the window operators use.
//!
//! THIS MODULE IS THE WHOLE OF THE ARITHMETIC HAT DOES OUTSIDE THE KERNELS, and
//! it exists because two of HAT's three window operators are *not* simple reshapes:
//!
//! * the local window attention runs on 16x16 windows, and the odd-numbered blocks
//!   roll the plane by `-window/2` before partitioning and by `+window/2` after, so
//!   the partition is taken at a CYCLIC offset. The reference writes the roll and
//!   the reshape as two operations; here both are folded into one index (`wi`),
//!   which is also what makes the shift free.
//! * the overlapping cross attention (OCAB) reads a 13x13 neighbourhood of a
//!   16x16-strided grid around each window, with the reference's `nn.Unfold`
//!   padding the plane with zeros. That is a second gather (`ki`) over a plane with
//!   a different stride, and it is the only place in the model where an input pixel
//!   is read by more than one output token.
//!
//! Everything here is host-side index arithmetic, identical on both backends: the
//! devices get the two index maps as device buffers and do the loads. That is
//! deliberate - it keeps one copy of the padding convention, the shift, and the
//! `Unfold` channel order, instead of one per backend.
//!
//! PADDING. The reference has no padding at all: `HAT.forward` reshapes h into
//! `h/16` window rows and raises if it does not divide, so it only ever runs on
//! window-multiple images. This engine takes any size and pads to the next window
//! multiple, which is the same thing the reference's own tiling script does before
//! calling the network - see `Plan::pad`. A window-multiple input is therefore
//! bit-for-bit the reference's geometry, and that is what the golden fixtures test.

/// One forward pass's shapes. All of them are derived from `(h, w)` and the
/// checkpoint, and none of them depend on the backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The input size, in pixels - what the caller asked for.
    pub h: usize,
    pub w: usize,
    /// The padded plane the network actually runs on: `h` and `w` rounded up to a
    /// window multiple.
    pub hp: usize,
    pub wp: usize,
    /// The embedding width.
    pub c: usize,
    /// The window side (16 for every released checkpoint).
    pub win: usize,
    /// `win / 2`: the roll applied by the odd-numbered blocks.
    pub shift: usize,
    /// The OCAB kernel's side: `window_size + int(overlap_ratio * window_size)`,
    /// which is `3 * win / 2` at the released `overlap_ratio` of 0.5.
    pub owin: usize,
}

impl Plan {
    pub fn new(h: usize, w: usize, win: usize, c: usize) -> Plan {
        let shift = win / 2;
        let owin = win + win / 2; // int(win * 0.5) + win, integer-exact at 16
        Plan {
            h,
            w,
            hp: h.div_ceil(win) * win,
            wp: w.div_ceil(win) * win,
            c,
            win,
            shift,
            owin,
        }
    }

    /// Pixels in the padded plane - one token each, since `patch_size` is 1.
    #[inline]
    pub fn tokens(&self) -> usize {
        self.hp * self.wp
    }

    /// Windows along x, windows along y, and the window count - which is
    /// `nx * ny`, NOT a square. A single `hp / win` side is right only for a square
    /// plane: 32x48 has 2 window rows and 3 window columns, so every map built from
    /// one side covers 4 of its 6 windows and the rest of the plane is never read.
    /// The reference has no such assumption (`h // 16` and `w // 16` are separate),
    /// and every fixture this engine had been verified against was square, so the
    /// bug only appears on a non-square input. `tests/plan.rs` checks the maps at
    /// 32x48 and 64x32 for exactly this reason.
    #[inline]
    pub fn nx(&self) -> usize {
        self.wp / self.win
    }
    #[inline]
    pub fn ny(&self) -> usize {
        self.hp / self.win
    }
    #[inline]
    pub fn nw(&self) -> usize {
        self.nx() * self.ny()
    }

    /// Tokens in one attention window: `win * win`.
    #[inline]
    pub fn wq(&self) -> usize {
        self.win * self.win
    }

    /// Key positions in one OCAB window: `owin * owin`.
    #[inline]
    pub fn wk(&self) -> usize {
        self.owin * self.owin
    }

    /// The OCAB unfold's padding: `(owin - win) / 2`, which is 3 at the released
    /// sizes. The window grid is `win`-strided, so a key column lies in
    /// `[win * window + ox - pad, ...)` of the plane.
    #[inline]
    pub fn opad(&self) -> usize {
        (self.owin - self.win) / 2
    }
}

/// The window partition's index map: for every window, in window-major order, the
/// padded-plane pixel each of its `win * win` tokens comes from.
///
/// `shift` is the CYCLIC offset the reference applies with `torch.roll(plane,
/// -shift, dims=(1, 2))` before partitioning. Folding it in here is what lets the
/// kernel read the plane once: a roll would be a separate full-plane copy per
/// block, and there are `shift > 0` blocks in every other HAB.
///
/// The wrap is over the WHOLE plane, not over each window, because that is what
/// `torch.roll` does - rolling by 8 moves the leftmost 8 columns of the plane to
/// the right edge, across window boundaries. That is the point of the shifted
/// blocks: after the roll, a window contains the right half of its own window and
/// the left half of its right neighbour, which is how HAT or Swin gets
/// cross-window information without a global attention.
pub fn window_index(plan: &Plan, shift: usize) -> Vec<u32> {
    let (win, wp, hp) = (plan.win, plan.wp, plan.hp);
    let (ny, nx) = (plan.ny(), plan.nx());
    let mut idx = Vec::with_capacity(plan.nw() * win * win);
    for wy in 0..ny {
        for wx in 0..nx {
            for y in 0..win {
                for x in 0..win {
                    // THE ROLL'S DIRECTION: `torch.roll(p, -shift)` is
                    // `out[i] = p[(i + shift) % n]`, not `p[(i - shift) % n]`. The
                    // two differ by a 2*shift offset and only one of them matches
                    // the reference - checked against `torch.roll` plus
                    // `window_partition` at 64x64, 48x48 and 32x48 in
                    // `tests/plan.rs`, which is where that pair is pinned down.
                    let src_y = (wy * win + y + shift) % hp;
                    let src_x = (wx * win + x + shift) % wp;
                    idx.push((src_y * wp + src_x) as u32);
                }
            }
        }
    }
    idx
}

/// The OCAB key gather's index map, and its zero-padding mask.
///
/// The reference is `nn.Unfold(kernel_size=13, stride=16, padding=3)` applied to a
/// `[b, 2c, hp, wp]` plane, so window `(wy, wx)`'s keys are the plane's
/// `[16*wy - 3, 16*wy + 10) x [16*wx - 3, 16*wx + 10)` rectangle, and positions
/// outside the plane read as ZERO. Only `3` pixels of context are real at an edge,
/// which is why the padding has to be represented rather than clamped: a clamp
/// would repeat the edge pixel, and the difference is visible in the output.
///
/// `idx[w][k] = usize::MAX` marks a zero position. The kernel writes 0.0 for it
/// instead of loading, so the sentinel never reaches a pointer.
///
/// CLAMP conventions: for a rectangle that lies entirely outside the plane the
/// index would be meaningless even as a sentinel, so the index stores the clamped
/// position and the mask says not to load. Keeping a valid index in every slot
/// means a kernel can compute an address unconditionally and select on the mask,
/// which is one compare instead of a branch on a possibly-negative row.
pub fn unfold_index(plan: &Plan) -> (Vec<u32>, Vec<u8>) {
    let (win, wp, hp, owin, pad) = (plan.win, plan.wp, plan.hp, plan.owin, plan.opad());
    let (ny, nx) = (plan.ny(), plan.nx());
    let n = owin * owin;
    let mut idx = Vec::with_capacity(plan.nw() * n);
    let mut keep = Vec::with_capacity(plan.nw() * n);
    for wy in 0..ny {
        for wx in 0..nx {
            for y in 0..owin {
                for x in 0..owin {
                    // `isize` because the rectangle starts at -3 for the first window
                    // and the test for "outside" has to be able to name that.
                    // Cast BEFORE the subtraction: the window grid is offset by
                    // `-pad`, and for the border windows an unsigned subtraction
                    // would wrap instead of going negative, silently turning an
                    // out-of-plane sample into a huge in-range index.
                    let sy = (wy * win + y) as isize - pad as isize;
                    let sx = (wx * win + x) as isize - pad as isize;
                    // Inside means both coordinates are in range: the negative half
                    // is checked first, because a negative isize cast to usize would
                    // compare as enormous.
                    let inside = sy >= 0 && sx >= 0 && (sy as usize) < hp && (sx as usize) < wp;
                    // `sy`/`sx` are isize because they go negative just below the
                    // plane's edge; the clamp brings them into [0, hp/wp) and the
                    // bounds have to be isize for it. The result is known
                    // non-negative, so the cast is a narrowing by construction.
                    let (cy, cx) = (sy.clamp(0, hp as isize - 1), sx.clamp(0, wp as isize - 1));
                    let (cy, cx) = (cy as usize, cx as usize);
                    idx.push((cy as usize * wp + cx as usize) as u32);
                    keep.push(inside as u8);
                }
            }
        }
    }
    (idx, keep)
}

/// The shifted-window attention mask, `[nw][n][n]`, as the reference computes it
/// for the odd-numbered blocks.
///
/// THE MASK DEPENDS ON THE IMAGE'S SIZE, AND IT IS NON-TRIVIAL AT EVERY SIZE THIS
/// ENGINE RUNS - WHICH IS THE OPPOSITE OF WHAT A FIRST READING SUGGESTS. The
/// reference slices each axis into `(0 .. -16)`, `(-16 .. -8)`, `(-8 ..)`, so at
/// 48x48 the cuts are `(0..32, 32..40, 40..48)` while the WINDOW grid is
/// `(0..16, 16..32, 32..48)`: the last window row and column straddle two labelled
/// rectangles, and the mask is non-zero there. Measured against the reference's own
/// `calculate_mask`: 114688 non-zero entries at 32x32, 180224 at 48x48, 245760 at
/// 64x64. The straddle only disappears when `win` divides the axis into regions
/// that line up with the cuts, i.e. when `len - win <= len - shift` is the whole
/// story - it is not, for any window-multiple length, because the cuts are offset
/// by `win - shift` from the grid. A version that assumed a uniform two-band split
/// (which is what a plain reading of Swin's mask suggests), or one that assumed an
/// all-zero mask for window-multiple inputs, would be wrong in a way that shows up
/// as a slightly wrong image at every size. `tests/plan.rs` compares this against
/// the reference's `calculate_mask` at six geometries for that reason.
///
/// THE FULL MASK IS `nw * win^4` FLOATS, WHICH IS THE LARGEST THING THIS ENGINE
/// ALLOCATES at large sizes - 67 MB at 256x256 and 1.07 GB at 1024x1024, against a
/// few hundred MB of activations. It is also highly redundant: the mask is a
/// function of the window's LABEL GRID, `win*win` u32, so the attention derives it
/// from that instead and `mask_labels` is what is stored. See `mask_label`.
///
/// The values are `0.0` and `-100.0`, both FINITE: adding -inf would turn a
/// masked-out row into a row of NaN in the softmax, and the reference's own
/// `torch.softmax` over an all-(-100) row is a uniform distribution, which is what
/// a finite mask reproduces.
///
/// The slice bounds are CLAMPED to the axis, so a slice that is empty in the
/// reference (a `slice(32, 40)` of a 32-row plane yields nothing) is simply
/// skipped: that is the same labelling the reference's in-place assignment
/// produces, where an empty slice writes nothing.
pub fn shift_mask(plan: &Plan) -> Vec<f32> {
    let (hp, wp, win, shift) = (plan.hp, plan.wp, plan.win, plan.shift);
    // The three slices per axis, as (start, end) with both clamped into range.
    let cuts = |len: usize| {
        [
            (0, len.saturating_sub(win)),
            (len.saturating_sub(win), len.saturating_sub(shift)),
            (len.saturating_sub(shift), len),
        ]
    };
    // Label every pixel of the padded plane with its region, then compare labels
    // inside each window - a transcription of the reference rather than a
    // derivation, so it cannot disagree with it about which geometry is which.
    let mut label = vec![0u32; hp * wp];
    let mut region = 0u32;
    for (y0, y1) in cuts(hp) {
        for (x0, x1) in cuts(wp) {
            for y in y0..y1 {
                for x in x0..x1 {
                    label[y * wp + x] = region;
                }
            }
            region += 1;
        }
    }
    // The mask is per WINDOW, in the window-major order `window_index` uses, and
    // is the same for every block that uses it - but not, as it happens, the same
    // for every window: a window at the plane's edge can straddle two regions when
    // the plane is odd in windows. Windows are half-open on the plane and always
    // inside it, so no clamping is needed here.
    let n = win * win;
    let (ny, nx) = (plan.ny(), plan.nx());
    let mut mask = Vec::with_capacity(plan.nw() * n * n);
    for wy in 0..ny {
        for wx in 0..nx {
            let mut lab = [0u32; 64 * 64];
            for y in 0..win {
                for x in 0..win {
                    lab[y * win + x] = label[(wy * win + y) * wp + wx * win + x];
                }
            }
            for q in 0..n {
                for k in 0..n {
                    mask.push(if lab[q] == lab[k] { 0.0 } else { -100.0 });
                }
            }
        }
    }
    mask
}

/// The mask as the LABEL GRID it is a function of: `[nw][win*win]` u32, which is
/// `win*win` times smaller than the mask itself (64 KB against 4 MB at 64x64, and
/// 4 MB against 1.07 GB at 1024x1024). The attention builds the additive mask from
/// this on the fly - one integer compare per score - which is what makes the largest
/// allocation in this engine proportional to `nw * win^2` instead of `nw * win^4`.
///
/// This is the reference's own decomposition: `calculate_mask` labels the plane,
/// partitions the labels, and then derives the mask from the labels.
pub fn mask_label(plan: &Plan) -> Vec<u32> {
    let (hp, wp, win, shift) = (plan.hp, plan.wp, plan.win, plan.shift);
    let cuts = |len: usize| {
        [
            (0, len.saturating_sub(win)),
            (len.saturating_sub(win), len.saturating_sub(shift)),
            (len.saturating_sub(shift), len),
        ]
    };
    let mut label = vec![0u32; hp * wp];
    let mut region = 0u32;
    for (y0, y1) in cuts(hp) {
        for (x0, x1) in cuts(wp) {
            for y in y0..y1 {
                for x in x0..x1 {
                    label[y * wp + x] = region;
                }
            }
            region += 1;
        }
    }
    let n = win * win;
    let (ny, nx) = (plan.ny(), plan.nx());
    let mut out = Vec::with_capacity(plan.nw() * n);
    for wy in 0..ny {
        for wx in 0..nx {
            for y in 0..win {
                for x in 0..win {
                    out.push(label[(wy * win + y) * wp + wx * win + x]);
                }
            }
        }
    }
    out
}

/// The per-RHAG maps and tables the attention needs, held once per plan.
///
/// The relative-position INDEX maps are the reference's registered buffers - they
/// are `torch.arange` arithmetic over the window geometry and do not depend on the
/// image or the weights, so they are built here rather than read from the
/// checkpoint. `tests/plan.rs` compares them against the checkpoint's copies, which
/// is what makes building them a checked decision instead of a re-derivation.
///
/// The BIAS tables are weights: `[npoints][nH]` per block, indexed by the maps. The
/// reference gathers `table[rpi.view(-1)]` on every forward; here the gather happens
/// in the attention loop, which needs no extra memory and no second copy of the
/// table. `--gather-bias` in `tools/convert.py` is the other side of that choice.
/// THE SIGN CONVENTION IS (query - key) HERE, AND (key - query) IN `rpi_oca`.
/// The reference builds the two tables with opposite subtraction orders
/// (`coords_flatten[:, :, None] - coords_flatten[:, None, :]` for SA against
/// `coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]` for OCA), and
/// the checkpoint's own index buffers are the only oracle for which is which: a
/// shared helper with one sign passes neither at both. Verified against
/// `relative_position_index_SA` of HAT-S_SRx4 on the host before this was written -
/// the key-minus-query form gives [480, 481, 482, 483] for the first four entries
/// where the checkpoint has [480, 479, 478, 477].
pub fn rpi_sa(win: usize) -> Vec<u32> {
    let n = win * win;
    let mut out = vec![0u32; n * n];
    for qy in 0..win {
        for qx in 0..win {
            for ky in 0..win {
                for kx in 0..win {
                    let dh = (qy + win - 1) as isize - ky as isize;
                    let dw = (qx + win - 1) as isize - kx as isize;
                    out[(qy * win + qx) * n + (ky * win + kx)] = (dh * (2 * win - 1) as isize + dw) as u32;
                }
            }
        }
    }
    out
}

/// The OCAB's index map: `[nq][nk]` over a 16-wide query grid and a 24-wide key
/// grid, with the reference's asymmetric axis order - `coords_ext[:, None] -
/// coords_ori[:, :, None]`, i.e. (key - query), and a shift of
/// `1 - (owin - win) = -7` on both axes.
///
/// THE RAW INDEX IS NEGATIVE AND THAT IS THE REFERENCE'S OWN CONVENTION. The
/// shift of -7 cannot make `key - query` non-negative when the key grid is wider
/// than the query grid: the difference spans `[-(win-1), owin-1] = [-15, 23]`, so
/// after +(-7) a component lies in `[-22, 16]`. The reference does not care, because
/// it gathers `relative_position_bias_table[rpi.view(-1)]` and PyTorch's advanced
/// indexing wraps a negative index from the END of the table - and the checkpoint's
/// own `relative_position_index_OCA` is stored with those negatives (measured: 256 x
/// 576 entries, min -880, max 640, and it matches this formula entry for entry).
///
/// A `u32` map cannot hold them, so each index is reduced into the table's own row
/// count, `(win + owin - 1)^2`, with `rem_euclid` - which is exactly Python's
/// negative-index wrap. The row selected is therefore the same row PyTorch would
/// gather, and the kernel's index stays unsigned.
///
/// The local-window map (`rpi_sa`) needs none of this: its shift of `win - 1` puts
/// every component in `[0, 2*win-2]` and the checkpoint's `relative_position_index_SA`
/// is correspondingly non-negative (min 0, max 960). The asymmetry between the two
/// maps is the reference's.
pub fn rpi_oca(win: usize, owin: usize) -> Vec<u32> {
    let nq = win * win;
    let nk = owin * owin;
    let shift = win as isize - owin as isize + 1;
    let mut out = vec![0u32; nq * nk];
    for qy in 0..win {
        for qx in 0..win {
            for ky in 0..owin {
                for kx in 0..owin {
                    let dh = ky as isize - qy as isize + shift;
                    let dw = kx as isize - qx as isize + shift;
                    let rows = ((win + owin - 1) * (win + owin - 1)) as isize;
                    out[(qy * win + qx) * nk + (ky * owin + kx)] =
                        (dh * (win + owin - 1) as isize + dw).rem_euclid(rows) as u32;
                }
            }
        }
    }
    out
}
