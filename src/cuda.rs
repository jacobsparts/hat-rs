//! The CUDA module layer: the two fatbins, and one typed launch per kernel this
//! engine uses.
//!
//! `build.rs` compiles lightgpu's `cuda/kernels.cu` into `hat_toolkit.fatbin` and
//! this crate's `cuda/hat.cu` into `hat_project.fatbin` and checks both name lists
//! against their sources in both directions. This module loads them and resolves a
//! launch by name, PROJECT FIRST (the two modules are separate, so a name can only
//! be in one of them - build.rs enforces that).
//!
//! Every helper exists so `gpu.rs` reads as the graph rather than as argument
//! lists. THE ARGUMENT ORDER IS THE ONE THING THE COMPILER CANNOT CHECK, which is
//! why each helper names its kernel and restates its shapes, and why
//! `--cuda-selftest` runs every project kernel against a host implementation of the
//! same operator. A mis-ordered argument produces a plausible image.
use lightgpu::vm::{self, Args, DevBuf, Launch, Module};

/// The block size for the elementwise and layout kernels. The toolkit indexes with
/// `blockIdx.x * blockDim.x + threadIdx.x`, so any size works; 256 is what its own
/// launch sites use and keeps the grid one dimensional.
pub const BLOCK: u32 = 256;

/// One block per 256 elements.
pub fn grid(n: usize) -> u32 {
    ((n as u64).div_ceil(BLOCK as u64)).max(1) as u32
}

pub struct Cuda {
    pub toolkit: Module,
    pub project: Module,
}

/// Per-KERNEL device timing, for localising a cost the way `dump` localises an error.
///
/// `HAT_RS_TIME=1` makes every launch record a pair of CUDA events around itself and
/// `report` print the device milliseconds accumulated per kernel name. It is the GPU
/// twin of `cpu::prof` and exists for the same reason: the only number the engine
/// otherwise has is the wall clock of a whole forward, which says nothing about
/// WHICH kernel to attack - and on this workload one or two kernels dominate.
///
/// THE EVENTS ARE RECORDED, NOT SYNCHRONISED. `cuEventRecord` only enqueues, so the
/// launches stay asynchronous and the pipeline is not serialised by the measurement;
/// one `cuCtxSynchronize` at report time resolves the whole queue at once. What is
/// attributed to a kernel is its DEVICE time, which excludes the host-side launch
/// cost - the right thing to optimize against, and the reason a profile can be
/// dominated by a kernel while the wall clock is dominated by launches (in which
/// case the report shows a total well below the measured forward, and the gap is the
/// answer rather than a defect in the measurement).
mod prof {
    use lightgpu::vm::Event;
    use std::cell::RefCell;
    use std::sync::OnceLock;

    fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| std::env::var("HAT_RS_TIME").map(|v| !v.is_empty()).unwrap_or(false))
    }

    /// One bracketed launch, holding its two events so both are destroyed with it.
    struct Rec {
        name: String,
        start: Event,
        end: Event,
    }

    /// THREAD LOCAL, NOT GLOBAL: a forward drives one stream from one thread, and a
    /// `CUevent` is not `Send` (it is a raw handle), so a process-wide list would not
    /// compile - and would not be the right model either, since two concurrent
    /// forwards would interleave their brackets into one meaningless total.
    thread_local! {
        static RECS: RefCell<Vec<Rec>> = const { RefCell::new(Vec::new()) };
    }

    /// Start bracketing a launch. `None` when the report is off, which is what keeps
    /// the instrumentation cheap in a normal build: one `OnceLock` lookup per launch,
    /// no event created and no allocation.
    pub fn mark() -> Option<(Event, Event)> {
        if !enabled() {
            return None;
        }
        let start = Event::new().ok()?;
        let end = Event::new().ok()?;
        start.record().ok()?;
        Some((start, end))
    }

    /// Finish bracketing: record the end event and file the pair under the name.
    pub fn done(name: &str, mark: Option<(Event, Event)>) {
        let (start, end) = match mark {
            Some(m) => m,
            None => return,
        };
        if end.record().is_err() {
            return;
        }
        RECS.with(|r| {
            if let Ok(mut v) = r.try_borrow_mut() {
                v.push(Rec { name: name.to_string(), start, end });
            }
        });
    }

    /// The number of launches bracketed since the last report, for the host-side
    /// question: a forward whose DEVICE time is well below its wall clock is either
    /// launch-bound or waiting on something that is not a kernel.
    pub fn counts_total() -> usize {
        let mut n = 0usize;
        RECS.with(|r| {
            if let Ok(v) = r.try_borrow() {
                n = v.len();
            }
        });
        n
    }

    /// Synchronise, sum by name, and print descending. Called at the end of every
    /// forward, so a tiled run reports one table per tile.
    pub fn report() {
        if !enabled() {
            return;
        }
        // The queue is resolved BEFORE the events are read, and before they are
        // dropped, so the elapsed times are real and not a queue of zeros.
        let n = RECS.with(|r| r.borrow().len());
        if n == 0 {
            return;
        }
        if lightgpu::vm::sync().is_err() {
            return;
        }
        // Counts as well as times: a kernel launched once and a kernel launched four
        // hundred times need different fixes even when their total time is the same.
        let mut by: Vec<(String, f64, usize)> = Vec::new();
        RECS.with(|r| {
            if let Ok(v) = r.try_borrow() {
                for rec in v.iter() {
                    let ms = rec.start.elapsed_ms(&rec.end).unwrap_or(0.0) as f64;
                    match by.iter_mut().find(|(n, _, _)| *n == rec.name) {
                        Some((_, acc, c)) => { *acc += ms; *c += 1; }
                        None => by.push((rec.name.clone(), ms, 1)),
                    }
                }
            }
        });
        RECS.with(|r| {
            if let Ok(mut v) = r.try_borrow_mut() {
                v.clear();
            }
        });
        by.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total: f64 = by.iter().map(|(_, v, _)| *v).sum();
        let n: usize = by.iter().map(|(_, _, c)| *c).sum();
        eprintln!("  gpu profile, {total:.1} ms of device time in {n} launches:");
        for (name, t, c) in by {
            eprintln!("    {name:<22} {t:>9.2} ms {c:>6}x  {:>5.1}%",
                      100.0 * t / total.max(1e-9));
        }
    }
}

impl Cuda {
    pub fn new() -> Result<Cuda, String> {
        vm::init()?;
        let toolkit =
            Module::load(include_bytes!(concat!(env!("OUT_DIR"), "/hat_toolkit.fatbin")))?;
        let project =
            Module::load(include_bytes!(concat!(env!("OUT_DIR"), "/hat_project.fatbin")))?;
        Ok(Cuda { toolkit, project })
    }

    /// Resolve a kernel, project first.
    pub fn module_of(&self, name: &str) -> Result<&Module, String> {
        if self.project.has(name) {
            Ok(&self.project)
        } else if self.toolkit.has(name) {
            Ok(&self.toolkit)
        } else {
            Err(format!("neither fatbin defines `{name}`"))
        }
    }

    pub fn run(&self, name: &str, g: (u32, u32, u32), b: (u32, u32, u32), shared: u32,
               args: &mut Args) -> Result<(), String> {
        let mark = prof::mark();
        let r = args.launch(self.module_of(name)?, name, Launch::new(g, b).shared(shared));
        prof::done(name, mark);
        r
    }

    /// One-dimensional launch: `n` elements, `BLOCK` threads.
    fn flat(&self, name: &str, n: usize, args: &mut Args) -> Result<(), String> {
        self.run(name, (grid(n), 1, 1), (BLOCK, 1, 1), 0, args)
    }

    // -----------------------------------------------------------------------
    // Convolutions and matmuls
    // -----------------------------------------------------------------------

    /// `lg_conv3x3_winograd(in, w, bias, out, c_in, c_out, h, wd, c_chunk, ocb, act, act_p)`.
    ///
    /// F(4,3) tiled: grid `(ceil(wd/16), ceil(h/16), ceil(c_out/ocb))`, block
    /// `(256,1,1)`, shared `c_chunk * 36 * (16 + ocb)` floats. The kernel's own doc
    /// is explicit that the caller must agree with it about `ocb` and `c_chunk` and
    /// that disagreement is SILENT, so both live here: `ocb = 16` is the maximum the
    /// kernel supports (`256 / WG4_NT`), and `c_chunk = 4` stages four input channels
    /// per pass, which is (4*36*32) = 4608 floats = 18 KB of shared memory and
    /// therefore two resident CTAs per SM. With `ocb = 16` a `c_out` of 144 or 180
    /// fills 9 or 12 blocks: the 180 case leaves 12 of 192 threads inactive in the
    /// last block of each tile, i.e. 6% of the kernel's slots, and that is cheaper
    /// than padding the channel count and reading the pad back.
    ///
    /// `act`: 0 = none, 1 = relu, 2 = leaky relu with slope `act_p`. THERE IS NO
    /// GELU: the CAB's activation is a separate `lg_gelu_erf`, because erf-GELU is a
    /// different op and not a mode of this one.
    #[allow(clippy::too_many_arguments)]
    pub fn conv3x3(&self, inp: &DevBuf, out: &DevBuf, w: &DevBuf, bias: &DevBuf,
                   ci: usize, co: usize, h: usize, wd: usize, act: i32, act_p: f32)
                   -> Result<(), String> {
        const OCB: usize = 16;
        const C_CHUNK: usize = 4;
        let g = ((wd.div_ceil(16)) as u32, (h.div_ceil(16)) as u32, (co.div_ceil(OCB)) as u32);
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(w.ptr).ptr(bias.ptr).ptr(out.ptr)
            .i32(ci as i32).i32(co as i32).i32(h as i32).i32(wd as i32)
            .i32(C_CHUNK as i32).i32(OCB as i32).i32(act).f32(act_p);
        // SHARED IS IN BYTES, NOT FLOATS. The kernel declares
        // `extern __shared__ float smem[]` with `tbuf` at its base and `ubuf` after
        // `c_chunk*NT*STRIDE` floats, so it needs `c_chunk*36*(16+ocb)` FLOATS -
        // 18432 bytes here. Passing the float count instead let every `ubuf` write run
        // 14 KB past the shared region, which the hardware reports as an illegal
        // address from a LATER kernel (the third one after it, in this engine).
        self.run("lg_conv3x3_winograd", g, (256, 1, 1),
                 (C_CHUNK * 36 * (16 + OCB) * 4) as u32, &mut a)
    }

    /// `lg_conv3x3s1p1(in, w, bias, out, c_in, c_out, h, wd)` - the direct kernel,
    /// used by `--cuda-selftest` as the reference the winograd transform is checked
    /// against.
    pub fn conv3x3_direct(&self, inp: &DevBuf, out: &DevBuf, w: &DevBuf, bias: &DevBuf,
                          ci: usize, co: usize, h: usize, wd: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(w.ptr).ptr(bias.ptr).ptr(out.ptr)
            .i32(ci as i32).i32(co as i32).i32(h as i32).i32(wd as i32);
        self.flat("lg_conv3x3s1p1", co * h * wd, &mut a)
    }

    /// `lg_conv1x1(in, w, bias, out, c_in, c_out, h, wd)` - one thread per output
    /// element. The channel attention's two 1x1s use this rather than `gemm_tiled`
    /// because their input is a single pixel (`hw = 1`) and their `c_in` is 6, which
    /// `gemm_tiled` cannot tile.
    pub fn conv1x1(&self, inp: &DevBuf, out: &DevBuf, w: &DevBuf, bias: &DevBuf,
                   ci: usize, co: usize, hw: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(w.ptr).ptr(bias.ptr).ptr(out.ptr)
            .i32(ci as i32).i32(co as i32).i32(1).i32(hw as i32);
        self.flat("lg_conv1x1", co * hw, &mut a)
    }

    /// `lg_f32_gemm_tiled(w, x, y, ne0, ne1, ncols)` - `W[ne1][ne0] @ x[ne0][ncols]`
    /// with BOTH sides in the ordinary ROW-MAJOR form, i.e. `x` is
    /// `[ncols][ne0]` and `y` is `[ncols][ne1]`.
    ///
    /// THAT IS NOT WHAT "PLANE" MEANS, and the distinction cost this backend a
    /// debugging session: the kernel reads its activation as `x + c * ne0` with `c`
    /// the COLUMN index and writes `y[c * ne1 + j]`, so its rows are the GEMM's
    /// columns - a Linear on a row-major `[tokens][c_in]` buffer, which is exactly
    /// what `lg_linear` computes and what the window tokens need. A Linear on an NCHW
    /// PLANE is a different op and is `lg_conv1x1` / `conv1x1` below.
    ///
    /// `ne0 % 4 == 0` is required; `ncols % 32 == 0` keeps the tiles full.
    pub fn gemm_tiled(&self, w: &DevBuf, x: &DevBuf, y: &DevBuf,
                      ne0: usize, ne1: usize, ncols: usize) -> Result<(), String> {
        let g = (ne1.div_ceil(64) as u32, ncols.div_ceil(32) as u32, 1);
        let mut a = Args::new();
        a.ptr(w.ptr).ptr(x.ptr).ptr(y.ptr)
            .i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
        self.run("lg_f32_gemm_tiled", g, (256, 1, 1), 0, &mut a)
    }

    /// `lg_conv1x1_rb(in, w, bias, out, c_in, c_out, h, wd)` - THE PLANAR LINEAR:
    /// `out[o][p] = bias[o] + sum_c w[o][c] * in[c][p]` for every pixel `p`, with the
    /// plane's `hw` the independent columns and the bias a per-output-channel vector.
    /// The 4x4 register-tiled variant of `lg_conv1x1` and bit-identical to it, so the
    /// two are interchangeable and this one is used wherever the shape is not tiny.
    ///
    /// THIS, NOT `lg_f32_gemm_tiled`, is what a checkpoint Linear on an NCHW plane
    /// is - the whole-model consequence of which is that the graph is planar through
    /// the CONVOLUTIONAL and NORM operators and token-major through the window ones,
    /// with `lg_linear` bridging the two at the attention.
    pub fn conv1x1_rb(&self, inp: &DevBuf, out: &DevBuf, w: &DevBuf, bias: &DevBuf,
                      ci: usize, co: usize, hw: usize) -> Result<(), String> {
        let g = (hw.div_ceil(64) as u32, co.div_ceil(64) as u32, 1);
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(w.ptr).ptr(bias.ptr).ptr(out.ptr)
            .i32(ci as i32).i32(co as i32).i32(1).i32(hw as i32);
        self.run("lg_conv1x1_rb", g, (16, 16, 1), 0, &mut a)
    }

    // -----------------------------------------------------------------------
    // Norms and activations
    // -----------------------------------------------------------------------

    /// `lg_channel_layer_norm(x, w, b, y, c, hw, eps)` - a LayerNorm over the
    /// CHANNEL axis of a `[c][hw]` plane, one thread per pixel.
    ///
    /// The GPU uses this for every norm in the model, which is what removes the
    /// plane<->token transposes the CPU backend needs: the reference normalises over
    /// the last axis of `[b, h*w, c]`, and on a plane that axis is the channel index.
    /// The toolkit accumulates `E[x^2] - mean^2` where the CPU subtracts the mean
    /// first - the same quantity to within fp32 rounding, and the one place the two
    /// backends' arithmetic is not identical (noted because the parity tolerance
    /// depends on knowing where the differences are).
    pub fn channel_layer_norm(&self, x: &DevBuf, w: &DevBuf, b: &DevBuf, y: &DevBuf,
                              c: usize, hw: usize, eps: f32) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(w.ptr).ptr(b.ptr).ptr(y.ptr)
            .i32(c as i32).i32(hw as i32).f32(eps);
        self.flat("lg_channel_layer_norm", hw, &mut a)
    }

    /// `lg_gelu_erf(x, y, n)` - `nn.GELU()`'s default erf form, which is what the
    /// reference's CAB uses.
    pub fn gelu(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(y.ptr).i32(n as i32);
        self.flat("lg_gelu_erf", n, &mut a)
    }

    /// `lg_sigmoid(x, y, n)`.
    pub fn sigmoid(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(y.ptr).i32(n as i32);
        self.flat("lg_sigmoid", n, &mut a)
    }

    /// `lg_relu(x, y, n)`.
    pub fn relu(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(y.ptr).i32(n as i32);
        self.flat("lg_relu", n, &mut a)
    }

    /// `lg_lrelu(x, y, slope, n)` - the head's `LeakyReLU(0.01)`.
    pub fn lrelu(&self, x: &DevBuf, y: &DevBuf, slope: f32, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(y.ptr).f32(slope).i32(n as i32);
        self.flat("lg_lrelu", n, &mut a)
    }

    // -----------------------------------------------------------------------
    // Elementwise and affine
    // -----------------------------------------------------------------------

    /// `lg_add(a, b, y, n)` -> `y = a + b`. `y` may alias `a` or `b`.
    pub fn add(&self, a_: &DevBuf, b: &DevBuf, y: &DevBuf, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(a_.ptr).ptr(b.ptr).ptr(y.ptr).i32(n as i32);
        self.flat("lg_add", n, &mut a)
    }

    /// `lg_add_scaled(a, b, out, n, scale)` -> `out = a + scale * b`, which is how
    /// the HAB's `shortcut + attn + conv_x * conv_scale` is built in the reference's
    /// order: `(shortcut + attn)` first, then `+ conv_x*scale`.
    pub fn add_scaled(&self, a_: &DevBuf, b: &DevBuf, out: &DevBuf, scale: f32, n: usize)
                      -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(a_.ptr).ptr(b.ptr).ptr(out.ptr).i64(n as i64).f32(scale);
        self.flat("lg_add_scaled", n, &mut a)
    }

    /// `lg_copy(x, y, n)`.
    pub fn copy(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(y.ptr).i64(n as i64);
        self.flat("lg_copy", n, &mut a)
    }

    /// `lg_scale(x, y, s, n)`.
    pub fn scale(&self, x: &DevBuf, y: &DevBuf, s: f32, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(y.ptr).f32(s).i32(n as i32);
        self.flat("lg_scale", n, &mut a)
    }

    /// `lg_channel_affine(in, out, scale, shift, c, hw)` - a per-channel affine over
    /// a plane. The GEMMs here fold their bias into the weights, so this is used
    /// only where a bias has to be added to an existing plane (see `hat_plane_block`,
    /// which does it while copying a channel range).
    pub fn channel_affine(&self, inp: &DevBuf, out: &DevBuf, scale: &DevBuf, shift: &DevBuf,
                          c: usize, hw: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(out.ptr).ptr(scale.ptr).ptr(shift.ptr)
            .i32(c as i32).i32(hw as i32);
        self.flat("lg_channel_affine", c * hw, &mut a)
    }

    /// `lg_channel_mean(x, out, c, hw)` - `AdaptiveAvgPool2d(1)` over a plane, one
    /// block per channel.
    pub fn channel_mean(&self, x: &DevBuf, out: &DevBuf, c: usize, hw: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(out.ptr).i32(c as i32).i32(hw as i32);
        // ONE THREAD-BLOCK PER CHANNEL, 1024 THREADS: the kernel reduces over a static
        // 1024-wide shared array and halves from 512, so a 256-thread block would leave
        // the upper half of the array out of the reduction and return a wrong mean. Its
        // doc is explicit that the block size is part of the contract.
        // grid = (c, 1, 1) and ANY block size: the kernel's reduction loads r[i] from
        // index i upward in blockDim strides and then runs a halving tree from 512 over
        // the full 1024 static slots, so a 256-thread block is handled and the op
        // table says so explicitly. (This was briefly changed to 1024 on a misreading;
        // the real fault at the time was elsewhere.)
        self.run("lg_channel_mean", (c as u32, 1, 1), (BLOCK, 1, 1), 0, &mut a)
    }

    /// `lg_channel_scale(in, s, out, c, hw)` - `out[ch] = in[ch] * s[ch]`, the
    /// channel attention's gate. It WRITES `out` rather than scaling in place, which
    /// is why the CAB needs the conv's output in a buffer of its own.
    pub fn channel_scale(&self, inp: &DevBuf, s: &DevBuf, out: &DevBuf, c: usize, hw: usize)
                         -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(s.ptr).ptr(out.ptr).i32(c as i32).i32(hw as i32);
        self.flat("lg_channel_scale", c * hw, &mut a)
    }

    // -----------------------------------------------------------------------
    // Windows
    // -----------------------------------------------------------------------

    /// `lg_window_gather(x, tok, nw, n, nww, win, hp, wp, c, shift, w0)`. `x` is a
    /// `[c][hp][wp]` PLANE, `tok` becomes `[nw][n][c]`, `nww` is the number of window
    /// COLUMNS (so a non-square grid divides correctly), and `shift` is the cyclic
    /// offset that folds `torch.roll(-shift)` into the index.
    ///
    /// THE INDEX IS MODULO-WRAPPED IN BOTH AXES, which matters for the caller: at
    /// `shift = 0` a position outside the plane wraps to the far edge rather than
    /// reading zero, and the reference's overlapping attention wants a ZERO there.
    /// `hat_plane_edges` is what makes those wrapped reads read zero - see `gpu.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn window_gather(&self, x: &DevBuf, tok: &DevBuf, nw: usize, n: usize, nww: usize,
                         win: usize, hp: usize, wp: usize, c: usize, shift: usize, w0: usize)
                         -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(tok.ptr).i32(nw as i32).i32(n as i32).i32(nww as i32).i32(win as i32)
            .i32(hp as i32).i32(wp as i32).i32(c as i32).i32(shift as i32).i32(w0 as i32);
        self.flat("lg_window_gather", nw * n * c, &mut a)
    }

    /// `lg_window_scatter(tok, x, ...)` - the inverse of `window_gather` at the same
    /// shift, WRITING the plane rather than accumulating into it. A caller that wants
    /// `x += attn` must copy `x` aside first, which is what the HAB does with its
    /// shortcut.
    #[allow(clippy::too_many_arguments)]
    pub fn window_scatter(&self, tok: &DevBuf, x: &DevBuf, nw: usize, n: usize, nww: usize,
                          win: usize, hp: usize, wp: usize, c: usize, shift: usize, w0: usize)
                          -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(tok.ptr).ptr(x.ptr).i32(nw as i32).i32(n as i32).i32(nww as i32).i32(win as i32)
            .i32(hp as i32).i32(wp as i32).i32(c as i32).i32(shift as i32).i32(w0 as i32);
        self.flat("lg_window_scatter", nw * n * c, &mut a)
    }

    // -----------------------------------------------------------------------
    // HAT's own kernels
    // -----------------------------------------------------------------------

    /// `hat_window_index(out, nw, nww, win, hp, wp, shift)` -> `[nw][win*win][2]` i32
    /// (y, x). The selftest holds this against `plan::window_index`.
    pub fn window_index(&self, out: &DevBuf, nw: usize, nww: usize, win: usize,
                        hp: usize, wp: usize, shift: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(out.ptr).i32(nw as i32).i32(nww as i32).i32(win as i32)
            .i32(hp as i32).i32(wp as i32).i32(shift as i32);
        self.flat("hat_window_index", nw * win * win, &mut a)
    }

    /// `hat_bias_gather(tab, idx, out, nq, nk, heads)` -> `[heads][nq][nk]`.
    pub fn bias_gather(&self, tab: &DevBuf, idx: &DevBuf, out: &DevBuf,
                       nq: usize, nk: usize, heads: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(tab.ptr).ptr(idx.ptr).ptr(out.ptr).i32(nq as i32).i32(nk as i32).i32(heads as i32);
        self.flat("hat_bias_gather", heads * nq * nk, &mut a)
    }

    /// `hat_mask_build(labels, nw, nww, win, hp, wp, shift)` -> `[nw][win*win]` i32,
    /// the reference's `calculate_mask` up to the comparison, and `win*win` times
    /// smaller than the additive mask it replaces.
    pub fn mask_build(&self, labels: &DevBuf, nw: usize, nww: usize, win: usize,
                      hp: usize, wp: usize, shift: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(labels.ptr).i32(nw as i32).i32(nww as i32).i32(win as i32)
            .i32(hp as i32).i32(wp as i32).i32(shift as i32);
        self.flat("hat_mask_build", nw * win * win, &mut a)
    }

    /// `hat_unfold_kv(plane, kw, vw, nw, nww, win, owin, pad, hp, wp, c)` - the
    /// overlapping attention's key/value windows from the 2c plane.
    #[allow(clippy::too_many_arguments)]
    pub fn unfold_kv(&self, plane: &DevBuf, kw: &DevBuf, vw: &DevBuf, nw: usize, nww: usize,
                     win: usize, owin: usize, pad: usize, hp: usize, wp: usize, c: usize)
                     -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(plane.ptr).ptr(kw.ptr).ptr(vw.ptr).i32(nw as i32).i32(nww as i32).i32(win as i32)
            .i32(owin as i32).i32(pad as i32).i32(hp as i32).i32(wp as i32).i32(c as i32);
        self.flat("hat_unfold_kv", nw * owin * owin * 2 * c, &mut a)
    }

    /// `hat_attention_d<D>(q, kw, vw, bias, labels, out, nw, nq, nk, heads, scale,
    /// has_bias, masked)` - ONE BLOCK PER (window, head), one thread per QUERY, with
    /// the head as the slow index.
    ///
    /// THE `d` IS IN THE KERNEL NAME, not an argument. `float acc[d]` needs `d` at
    /// compile time to live in registers: with `d` a runtime argument the compiler
    /// reserved the worst case and the kernel reported a 256-byte stack, i.e. the
    /// accumulator in LOCAL MEMORY. The two instantiations cover every head_dim the
    /// family has (24 = 144/6, 30 = 180/6), and an unknown one is an error at launch
    /// rather than a silent fallback to the wrong size.
    ///
    /// `bias` is `[heads][nk][nq]` - TRANSPOSED relative to the natural
    /// `[heads][nq][nk]` - so that a warp's bias reads are consecutive. See
    /// `hat_bias_gather`.
    ///
    /// `bias` is the gathered `[heads][nq][nk]` table and may be null when
    /// `has_bias` is false, which is the overlapping attention: the reference runs it
    /// with no relative-position bias, and a null pointer saves a 3.5 MB zeroed table
    /// nothing would read. `labels` is the mask's label grid (null with `masked`
    /// false). One kernel serves both attentions because they differ only in `nq`/`nk`
    /// and in which of the two masks they have.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, q: &DevBuf, k: &DevBuf, v: &DevBuf, bias: Option<&DevBuf>,
                     labels: Option<&DevBuf>, out: &DevBuf, nw: usize, nww: usize,
                     nq: usize, nk: usize, heads: usize, d: usize, scale: f32,
                     has_bias: bool, masked: bool) -> Result<(), String> {
        let mut a = Args::new();
        // `nww` is now unused by the kernel: `blockIdx.x / heads` is the window.
        let _ = nww;
        a.ptr(q.ptr).ptr(k.ptr).ptr(v.ptr)
            .ptr(bias.map(|b| b.ptr).unwrap_or(0))
            .ptr(labels.map(|l| l.ptr).unwrap_or(0)).ptr(out.ptr)
            // NO `d`: it is in the kernel NAME (see above), and leaving it here
            // shifted every remaining argument by one - the int `d` landed in
            // `scale` and the float `scale` in `has_bias`, which is silent garbage
            // rather than a launch failure.
            .i32(nw as i32).i32(nq as i32).i32(nk as i32)
            .i32(heads as i32).f32(scale)
            .i32(has_bias as i32).i32(masked as i32);
        // grid (nw * heads, 1, 1), block (nq, 1, 1): the block spans queries so the
        // whole warp reads one key row, and `blockIdx.x` selects (window, head).
        let _ = d;
        let name = match d {
            24 => "hat_attention_d24",
            30 => "hat_attention_d30",
            other => return Err(format!(
                "head_dim {other}: no templated attention kernel (the family uses 24 and 30)"
            )),
        };
        self.run(name, ((nw * heads) as u32, 1, 1), (nq as u32, 1, 1), 0, &mut a)
    }

    /// `hat_plane_bias(plane, bias, c, hw)` - `plane[ch][p] += bias[ch]`, in place, the
    /// bias a `gemm_tiled` cannot supply.
    pub fn plane_bias(&self, plane: &DevBuf, bias: &DevBuf, c: usize, hw: usize)
                      -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(plane.ptr).ptr(bias.ptr).i32(c as i32).i32(hw as i32);
        self.flat("hat_plane_bias", c * hw, &mut a)
    }

    /// `hat_token_range(dst, src, co, ci, c0, rows)` - `dst[r][ch] = src[r][c0+ch]`,
    /// the token-major twin of `hat_plane_block`, for splitting the window attention's
    /// fused `[rows][3c]` qkv projection into q, k and v.
    pub fn token_range(&self, dst: &DevBuf, src: &DevBuf, co: usize, ci: usize, c0: usize,
                       rows: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(dst.ptr).ptr(src.ptr).i32(co as i32).i32(ci as i32).i32(c0 as i32)
            .i32(rows as i32);
        self.flat("hat_token_range", rows * co, &mut a)
    }

    /// `lg_linear(x, w, bias, out, rows, c_in, c_out)` - `out = x W^T + bias` for a
    /// ROW-major `[rows][c_in]` input. The window-token projections (the HAB's `qkv`,
    /// and `proj` on the attention's output) are row-major and so use this rather than
    /// `gemm_tiled`, whose input is a plane.
    pub fn linear(&self, x: &DevBuf, out: &DevBuf, w: &DevBuf, bias: &DevBuf,
                  ci: usize, co: usize, rows: usize) -> Result<(), String> {
        // `lg_linear` IS A 2-D KERNEL and its shape is part of its contract: block
        // (16,16,1) with `blockIdx.x` over output channels in 16s and `blockIdx.y`
        // over rows in 16s, so `threadIdx.y` is the row and `tx` the output. Launching
        // it flat makes every thread a row-0 lane and lets `blockIdx.x` run to
        // `rows*co/256`, i.e. it writes far past the output buffer - which the hardware
        // reports as an illegal address in a later launch, not there.
        let g = (co.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1);
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(w.ptr).ptr(bias.ptr).ptr(out.ptr)
            .i32(rows as i32).i32(ci as i32).i32(co as i32);
        self.run("lg_linear", g, (16, 16, 1), 0, &mut a)
    }

    /// `hat_plane_block(dst, src, bias, co, ci, c0, hw)` - copy channels
    /// `[c0, c0+co)` of `src` to `dst`, adding that range of `bias` (null for none).
    /// A fused projection writes one `[3c][hw]` plane and the reference's q, k, v and
    /// `cat(k, v)` are exactly channel ranges of it.
    pub fn plane_block(&self, dst: &DevBuf, src: &DevBuf, bias: Option<&DevBuf>,
                       co: usize, ci: usize, c0: usize, hw: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(dst.ptr).ptr(src.ptr).ptr(bias.map(|b| b.ptr).unwrap_or(0))
            .i32(co as i32).i32(ci as i32).i32(c0 as i32).i32(hw as i32);
        self.flat("hat_plane_block", co * hw, &mut a)
    }

    /// `hat_plane_edges(plane, c, hp, wp, pad)` - zero the `pad`-wide border rows and
    /// columns of every channel. See the kernel's note: the window gather wraps
    /// out-of-range reads to the far edge, and zeroing that edge is how a wrapped read
    /// becomes the zero the reference's padded unfold would have produced.
    pub fn plane_edges(&self, plane: &DevBuf, c: usize, hp: usize, wp: usize, pad: usize)
                       -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(plane.ptr).i32(c as i32).i32(hp as i32).i32(wp as i32).i32(pad as i32);
        self.flat("hat_plane_edges", c * hp * wp, &mut a)
    }

    /// `hat_oca_label(labels, nw, nww, win, owin, pad, hp, wp)` -> `[nw][owin*owin]`
    /// i32, 0 for a real key and 9 for a key the unfold reached through its zero
    /// padding. The attention's mask then removes every padded key from every query,
    /// which is what the reference's zero-padded unfold needs and what a bias of zero
    /// at those slots would otherwise have to guarantee.
    #[allow(clippy::too_many_arguments)]
    pub fn oca_label(&self, labels: &DevBuf, nw: usize, nww: usize, win: usize,
                     owin: usize, pad: usize, hp: usize, wp: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(labels.ptr).i32(nw as i32).i32(nww as i32).i32(win as i32)
            .i32(owin as i32).i32(pad as i32).i32(hp as i32).i32(wp as i32);
        self.flat("hat_oca_label", nw * owin * owin, &mut a)
    }

    /// `hat_pixel_shuffle(src, dst, c, h, w, r)` - `nn.PixelShuffle(r)`.
    ///
    /// `r` is 2 for the head's `Upsample` octaves and 3 for its single scale-3
    /// block; both come from `Weights::up_blocks()`. The toolkit has no shuffle in
    /// this direction (`lg_pixel_unshuffle2` is the space-to-depth twin and
    /// `lg_merge_2x2` is the ViT patch merge, the opposite permutation).
    pub fn pixel_shuffle(&self, src: &DevBuf, dst: &DevBuf, c: usize, h: usize, w: usize,
                         r: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(src.ptr).ptr(dst.ptr).i32(c as i32).i32(h as i32).i32(w as i32).i32(r as i32);
        self.flat("hat_pixel_shuffle", c * h * r * w * r, &mut a)
    }
}

/// Upload an i32 buffer (the checkpoint's relative-position index maps, which the
/// toolkit would otherwise have to grow an i32 accessor for). The bytes are written
/// little-endian, which is what the device reads on every architecture this targets.
pub fn upload_i32(values: &[i32]) -> Result<DevBuf, String> {
    let buf = DevBuf::alloc(values.len().max(1) * 4)?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    vm::copy_htod(buf.ptr, &bytes)?;
    Ok(buf)
}

/// Print the per-kernel device-time profile accumulated since the last report.
/// A no-op unless `HAT_RS_TIME` is set, so callers do not need to ask.
pub fn report() {
    prof::report();
}

/// Every kernel name this engine launches, checked against the loaded modules
/// before the first allocation so a missing kernel is one clear error rather than a
/// failure in the middle of a forward.
pub fn launched() -> &'static [&'static str] {
    &[
        "lg_conv3x3_winograd", "lg_conv3x3s1p1", "lg_conv1x1", "lg_conv1x1_rb",
        "lg_f32_gemm_tiled",
        "lg_linear",
        "lg_channel_layer_norm", "lg_gelu_erf", "lg_sigmoid", "lg_relu", "lg_lrelu",
        "lg_add", "lg_add_scaled", "lg_copy", "lg_scale", "lg_channel_affine",
        "lg_channel_mean", "lg_channel_scale", "lg_window_gather", "lg_window_scatter",
        "hat_window_index", "hat_bias_gather", "hat_mask_build", "hat_unfold_kv",
        "hat_attention_d24", "hat_attention_d30",
        "hat_plane_block", "hat_plane_edges", "hat_oca_label",
        "hat_pixel_shuffle", "hat_token_range", "hat_plane_bias",
    ]
}

/// Is a device present and does every kernel resolve?
pub fn available(cuda: &Cuda) -> Result<(), String> {
    for k in launched() {
        cuda.module_of(k)?;
    }
    Ok(())
}
