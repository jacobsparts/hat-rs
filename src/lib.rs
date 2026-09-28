//! hat - HAT image super-resolution as one binary.
//!
//! HAT (Hybrid Attention Transformer, XPixelGroup) is a Swin-style window
//! transformer with two changes that matter for a hand-written engine: every
//! window-attention block also carries a small convolutional channel-attention
//! branch on the block input, and each stage ends with an OVERLAPPING cross
//! attention whose keys come from a larger, strided window (`nn.Unfold`) than its
//! queries. The two share the head of SwinIR (pixelshuffle).
//!
//! There are two backends and neither is the correctness standard:
//!
//! * `Cpu` is pure Rust and parallelised with rayon. It is the path for a machine
//!   with no usable GPU, so its speed is a requirement and not a courtesy - see
//!   the "Parallelism" note at the top of `cpu.rs`.
//! * `Gpu` is CUDA, through the `lightgpu` toolkit's kernels plus this engine's
//!   own attention and unfold kernels in `cuda/hat.cu`.
//!
//! They are checked against the PUBLISHED PyTorch network: `tools/make_fixture.py`
//! runs the upstream `hat_arch.py` with a released configuration and released
//! weights, and `--verify` compares a backend's output with what it produced. A
//! backend that matches the other backend proves only that they share a mistake.
//! `--cuda-selftest` then checks each launched kernel against a host twin (see
//! `selftest.rs`), which is the only thing that catches a kernel whose result is
//! merely plausible: the four project kernels that do not exist in the toolkit all
//! take long shape argument lists, and a mis-ordered argument produces a plausible
//! image rather than a fault.
pub mod backend;
pub mod cpu;
pub mod dump;
pub mod fixture;
pub mod image;
pub mod plan;
pub mod tile;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod selftest;

/// The version string the banner prints, so a bug report can say which build.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
