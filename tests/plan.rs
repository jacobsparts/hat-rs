//! The plan's geometry against the reference's own operations.
//!
//! `src/plan.rs` builds the three host-side index structures HAT needs - the window
//! map (with the odd blocks' roll folded in), the OCAB's `nn.Unfold` gather, and the
//! shifted-window attention mask - and each of them is a convention that cannot be
//! derived from the doc comments. `tools/make_plan_oracle.py` records what PyTorch
//! actually does for six geometries (through `torch.roll` + `window_partition`,
//! `nn.Unfold`, and the reference's own `calculate_mask`), and this test compares the
//! engine against that record. Nothing here needs a checkpoint: it is pure geometry,
//! so it runs in a build with no model and no GPU.
//!
//! The index maps and the unfold's padding: the reference pads the plane with ZEROS
//! and `nn.Unfold` reads a zero from outside, while the engine stores the CLAMPED
//! in-plane position and relies on the `keep` byte to skip the load. The two
//! therefore disagree about the index of an out-of-plane slot by construction, and
//! agree about which slots those are - so `unfold_keep` is compared exactly and
//! `unfold_index` only where it is set.
use hat::plan::{rpi_oca, rpi_sa, shift_mask, unfold_index, window_index, Plan};

/// The oracle container: `HATPLAN1`, a u32 count, then per entry a u32 name length,
/// the name, a u8 dtype, a u32 rank, the rank's dimensions, and a u32 payload.
struct Oracle {
    names: Vec<String>,
    data: Vec<(Vec<usize>, Vec<u32>)>,
}

impl Oracle {
    fn load(path: &str) -> Oracle {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(&b[..8], b"HATPLAN1", "{path}: wrong magic");
        let mut off = 8usize;
        let count = le32(&b, off) as usize;
        off += 4;
        let mut names = Vec::new();
        let mut data = Vec::new();
        for _ in 0..count {
            let nl = le32(&b, off) as usize;
            off += 4;
            let name = std::str::from_utf8(&b[off..off + nl]).expect("name is utf8").to_string();
            off += nl;
            assert_eq!(b[off], 1, "{name}: the oracle holds u32 arrays only");
            off += 1;
            let rank = le32(&b, off) as usize;
            off += 4;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(le32(&b, off) as usize);
                off += 4;
            }
            let n: usize = dims.iter().product();
            let payload: Vec<u32> = (0..n).map(|i| le32(&b, off + 4 * i)).collect();
            off += 4 * n;
            names.push(name);
            data.push((dims, payload));
        }
        Oracle { names, data }
    }

    fn get(&self, name: &str) -> &(Vec<usize>, Vec<u32>) {
        let i = self.names.iter().position(|n| n == name)
            .unwrap_or_else(|| panic!("the oracle has no {name}; regenerate with tools/make_plan_oracle.py"));
        &self.data[i]
    }

    /// The mask the reference's `calculate_mask` produces, from the label grid it
    /// stores: `-100` where two tokens are in different labelled rectangles.
    fn mask_from_labels(&self, name: &str, win: usize) -> Vec<f32> {
        let (_, labels) = self.get(name);
        let n = win * win;
        let nw = labels.len() / n;
        let mut mask = Vec::with_capacity(nw * n * n);
        for w in 0..nw {
            let lab = &labels[w * n..(w + 1) * n];
            for q in 0..n {
                for k in 0..n {
                    mask.push(if lab[q] == lab[k] { 0.0 } else { -100.0 });
                }
            }
        }
        mask
    }
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

const WIN: usize = 16;
const C: usize = 4; // the plan's channel count does not affect the geometry

fn geometries() -> Vec<(usize, usize)> {
    vec![(16, 16), (32, 32), (48, 48), (32, 48), (64, 64), (64, 32)]
}

#[test]
fn the_window_map_matches_torch_roll_and_window_partition() {
    let o = Oracle::load("tests/data/plan_oracle.bin");
    for (h, w) in geometries() {
        let plan = Plan::new(h, w, WIN, C);
        for (shift, name) in [(0, "window_index_plain"), (plan.shift, "window_index_shifted")] {
            let (dims, want) = o.get(&format!("{h}x{w}/{name}"));
            let got = window_index(&plan, shift);
            assert_eq!(dims.iter().product::<usize>(), got.len(), "{h}x{w} {name}: length");
            assert_eq!(got, *want, "{h}x{w} {name}: the window map differs (shift {shift})");
        }
    }
}

#[test]
fn the_unfold_map_and_its_padding_mask_match_nn_unfold() {
    let o = Oracle::load("tests/data/plan_oracle.bin");
    for (h, w) in geometries() {
        let plan = Plan::new(h, w, WIN, C);
        let (idx, keep) = unfold_index(&plan);
        let (_, want_keep) = o.get(&format!("{h}x{w}/unfold_keep"));
        let (_, want_idx) = o.get(&format!("{h}x{w}/unfold_index"));
        assert_eq!(idx.len(), want_idx.len(), "{h}x{w}: unfold length");
        assert_eq!(
            keep.iter().map(|&k| k as u32).collect::<Vec<_>>(),
            *want_keep,
            "{h}x{w}: the unfold's padding mask differs"
        );
        // The index only means anything where the gather is live; everywhere else
        // the engine stores a clamped position and the reference reads a zero.
        let mut checked = 0usize;
        for i in 0..idx.len() {
            if want_keep[i] == 1 {
                assert_eq!(idx[i], want_idx[i], "{h}x{w}: unfold index {i} (live) differs");
                checked += 1;
            }
        }
        assert!(checked > 0, "{h}x{w}: no live unfold positions to compare");
    }
}

#[test]
fn the_shift_mask_matches_the_references_calculate_mask() {
    let o = Oracle::load("tests/data/plan_oracle.bin");
    let mut nonzero_seen = 0usize;
    for (h, w) in geometries() {
        let plan = Plan::new(h, w, WIN, C);
        let want = o.mask_from_labels(&format!("{h}x{w}/mask_labels"), WIN);
        let got = shift_mask(&plan);
        assert_eq!(got.len(), want.len(), "{h}x{w}: mask length");
        for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, e, "{h}x{w}: mask entry {i} differs ({} vs {e})", g);
        }
        nonzero_seen += want.iter().filter(|&&v| v != 0.0).count();
    }
    // A mask implementation that returned all zeros would pass every geometry where
    // the slices line up with the window grid - which is most of them - so the test
    // has to see at least one geometry where the mask actually masks something.
    assert!(nonzero_seen > 0, "no geometry produced a non-trivial shift mask");
}

#[test]
fn the_plan_pads_to_the_next_window_multiple() {
    assert_eq!(Plan::new(48, 48, WIN, C).hp, 48);
    assert_eq!(Plan::new(33, 17, WIN, C).hp, 48);
    assert_eq!(Plan::new(33, 17, WIN, C).wp, 32);
    let p = Plan::new(48, 48, WIN, C);
    assert_eq!(p.tokens(), 2304);
    assert_eq!(p.nw(), 9);
    assert_eq!(p.wq(), 256);
    assert_eq!(p.wk(), 576);
    assert_eq!(p.opad(), 4);
    assert_eq!(p.nx(), 3);
    assert_eq!(p.ny(), 3);
    // The plan keeps the INPUT size as well as the padded one, so two inputs that
    // pad to the same geometry are different plans with the same working shapes -
    // the cache keys on the whole struct, and the padded fields are what the
    // arithmetic uses.
    assert_ne!(Plan::new(33, 33, WIN, C), Plan::new(48, 48, WIN, C));
    let (a, b) = (Plan::new(33, 33, WIN, C), Plan::new(48, 48, WIN, C));
    assert_eq!((a.hp, a.wp, a.nw(), a.tokens()), (b.hp, b.wp, b.nw(), b.tokens()));
    assert_ne!(Plan::new(33, 33, WIN, C), Plan::new(33, 32, WIN, C));
    // The window count is per-axis: 32x48 is 2 x 3, not a square of either.
    let rect = Plan::new(32, 48, WIN, C);
    assert_eq!((rect.ny(), rect.nx(), rect.nw()), (2, 3, 6));
    assert_eq!(Plan::new(48, 32, WIN, C).nw(), 6);
    assert_eq!(Plan::new(48, 32, WIN, C).ny(), 3);
}

/// The two relative-position index maps are `arange` arithmetic over the window
/// geometry and are shipped as buffers in the checkpoint, so the checkpoint's copy is
/// the oracle - no torch run needed. This is the same comparison `tests/parity.rs`
/// makes, kept here as well because it is the plan's convention, not the backend's.
#[test]
fn the_index_maps_are_the_ones_the_reference_registers() {
    let win = WIN;
    let sa = rpi_sa(win);
    assert_eq!(sa.len(), win * win * win * win);
    assert_eq!(sa[0], ((win - 1) * (2 * win - 1) + (win - 1)) as u32);
    // rpi_sa's first row walks the key leftwards, so it decreases by one.
    assert_eq!(&sa[..4], &[480, 479, 478, 477]);
    let oca = rpi_oca(win, win + win / 2);
    assert_eq!(oca.len(), win * win * 576);
    // The OCA map is (key - query) with the opposite sign, so its first row walks
    // the 24-wide key grid and increases.
    assert_eq!(&oca[..4], &[1241, 1242, 1243, 1244]);
    // The table has (win + owin - 1)^2 rows: 39^2 = 1521 at the released sizes.
    let owin = win + win / 2;
    let rows = (win + owin - 1) * (win + owin - 1);
    assert!(oca.iter().all(|&i| (i as usize) < rows), "an OCA row is out of the table");
}
