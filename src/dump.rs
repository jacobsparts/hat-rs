//! Activation dumps, shared by both backends.
//!
//! `HAT_RS_DUMP=<dir>` makes a forward write every activation it computes as
//! `<dir>/<name>.f32`, named by the reference's own module path with dots replaced
//! by underscores - the same convention `tools/make_fixture.py --dump` uses. That
//! convention is the whole point: when two backends disagree, comparing their
//! dumps by name localises the divergence to ONE operator, where comparing output
//! images localises it to a stage at best. It has already earned its place once -
//! the CUDA backend's first wrong result was found this way.
//!
//! The format is the fixture's: a u32 dimension count, that many u32 dimensions,
//! then little-endian f32 data.
/// Activation dumps, for localising a divergence against the reference.
///
/// `HAT_RS_DUMP=<dir>` makes the forward write every activation it computes as
/// `<dir>/<name>.f32`, where the name is the reference's own module path with dots
/// replaced by underscores - the same convention `tools/make_fixture.py --dump`
/// uses - so the two can be compared stage by stage instead of by bisecting the
/// output image. It is a debugging aid that costs one `OnceLock` lookup per call
/// when it is off, and it exists because a wrong activation is much easier to find
/// than a wrong image.
///
/// THE FORMAT IS A CLONE OF THE FIXTURE'S, for the same reason: a u32 dimension
/// count, that many u32 dimensions, then the f32 data little-endian. `xxd -e` reads
/// it and so does a 5-line Python function, and neither needs numpy on the Rust
/// side or a torch on the Python side.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

fn dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = std::env::var("HAT_RS_DUMP").ok().filter(|s| !s.is_empty())?;
        let p = PathBuf::from(d);
        std::fs::create_dir_all(&p).ok()?;
        Some(p)
    })
    .as_ref()
}

/// Is dumping on?
///
/// CALL THIS BEFORE GATHERING THE DATA, not after: on the GPU the gather is a
/// `cuMemcpyDtoH`, which SYNCHRONISES with the device, so a backend that downloads
/// an activation and then asks `write` whether to save it pays a pipeline flush per
/// dump point - thirty of them per forward here, which was worth ~300 ms of a
/// 467 ms pass. The CPU pays a copy for the same reason.
pub fn enabled() -> bool {
    dir().is_some()
}

/// Write `data` as `name` with the given dimensions. A failure is REPORTED on
/// stderr and swallowed: a debugging aid must never take down the run it is
/// meant to explain.
pub fn write(name: &str, dims: &[usize], data: &[f32]) {
    let Some(dir) = dir() else { return };
    let n: usize = dims.iter().product();
    if n != data.len() {
        eprintln!("dump: {name}: {dims:?} is {n} floats but the data is {}", data.len());
        return;
    }
    let path = dir.join(format!("{name}.f32"));
    let mut buf = Vec::with_capacity(4 + 4 * dims.len() + 4 * data.len());
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        buf.extend_from_slice(&(*d as u32).to_le_bytes());
    }
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    match std::fs::File::create(&path).and_then(|mut f| f.write_all(&buf)) {
        Ok(()) => {}
        Err(e) => eprintln!("dump: {}: {e}", path.display()),
    }
}

/// Write a `[c][h][w]` plane.
pub fn plane(name: &str, plane: &[f32], c: usize, h: usize, w: usize) {
    if dir().is_none() {
        return;
    }
    let hw = h * w;
    write(name, &[c, h, w], &plane[..c * hw]);
}

/// The reference's module path -> the dump's file stem: dots become
/// underscores, which is the convention `tools/make_fixture.py --dump` uses.
/// Keeping the translation in one place is what lets both sides be paired by
/// name without a lookup table.
pub fn stem(path: &str) -> String {
    path.replace('.', "_")
}

/// Write a `[h][w][c]` token layout.
pub fn tokens(name: &str, tok: &[f32], c: usize, h: usize, w: usize) {
    if dir().is_none() {
        return;
    }
    write(name, &[h, w, c], &tok[..h * w * c]);
}
