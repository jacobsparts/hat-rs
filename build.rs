//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/hat.cu` (PROJECT_KERNELS). Each gets its own fatbin and its own
//! `--entries` list, and `src/cuda.rs` loads them as separate modules, so neither
//! can shadow a name in the other.
//!
//! Both lists are checked against the source they are compiled from before nvcc
//! runs, in BOTH directions: a name that is listed but not defined fails the
//! build, and so does a name that is defined but not listed - the latter would be
//! pruned from the fatbin and fail at launch instead.

/// The toolkit's kernels this engine launches.
///
/// HAT is mostly 3x3 convolutions at 144 or 180 channels (`conv_first`, each
/// stage's `conv`, `conv_after_body`, the CAB blocks and the whole reconstruction
/// head), which is exactly the regime F(4,3) Winograd is for: 36 products per 16
/// outputs against the direct kernel's 144. `lg_conv3x3s1p1` is launched by
/// NOTHING and is kept because `--cuda-selftest` checks the winograd transform
/// against it - a transform this engine got wrong is otherwise only visible as a
/// slightly wrong image.
///
/// `lg_conv1x1`/`lg_conv1x1_rb` are the CAB's two 1x1 convolutions and the
/// channel attention's squeeze/excite pair; `lg_linear`/`lg_linear_rb` are qkv,
/// the attention output projection and both MLP layers; `lg_layer_norm` is
/// `patch_embed.norm`, every block's `norm1`/`norm2`, the final `norm` and the
/// OCAB's pair; `lg_gelu_erf` is nn.GELU's default (erf, not tanh); `lg_add`,
/// `lg_add_scaled` and `lg_scale` are the block residuals; `lg_channel_mean`,
/// `lg_channel_scale` and `lg_sigmoid` are the channel attention's pooling and
/// gating; `lg_copy` is the stage input's copy into the residual buffer.
///
/// `lg_window_gather`/`lg_window_scatter` are the window partition/unpartition
/// pair, which is where the shifted-window `torch.roll` is folded into the gather
/// index (the toolkit pair takes a modulo-wrapped index map).
pub const TOOLKIT_KERNELS: &[&str] = &[
    "lg_conv3x3s1p1",
    "lg_conv3x3_winograd",
    "lg_conv1x1",
    "lg_conv1x1_rb",
    "lg_linear",
    "lg_f32_gemm_tiled",
    "lg_layer_norm",
    "lg_channel_layer_norm",
    "lg_channel_affine",
    "lg_channel_mean",
    "lg_channel_scale",
    "lg_sigmoid",
    "lg_gelu_erf",
    "lg_relu",
    "lg_lrelu",
    "lg_add",
    "lg_add_scaled",
    "lg_scale",
    "lg_copy",
    "lg_window_gather",
    "lg_window_scatter",
];

/// This engine's own kernels, in `cuda/hat.cu`.
///
/// The windowed attention is the architecture, not a generic op: the score pass
/// carries HAT's relative-position bias, its shifted-window mask and its per-head
/// scale, and the OCAB form applies a 13x13 strided unfold to the key/value plane
/// before it. The score/softmax misalignment across a warp is a property of the
/// window size and head count, and the two-window forms differ only in `nq`/`nk`.
///
/// `hat_plane_block` is the only layout kernel the backend needs: the reference's
/// fused `Linear(c, 3c)` writes one `[3c][hw]` plane and q, k, v and `cat(k, v)`
/// are CHANNEL RANGES of it, with the bias a range of the same fused vector - so
/// there is no split kernel, no derived weight and no plane<->token transpose
/// anywhere in the forward (every projection is `lg_f32_gemm_tiled` on a plane and
/// every norm is `lg_channel_layer_norm`). `hat_pixel_shuffle` is separate because
/// the toolkit's `lg_merge_2x2` is the ViT patch MERGE - four spatial neighbours
/// concatenated onto the channel axis - which is the opposite direction.
const PROJECT_KERNELS: &[&str] = &[
    "hat_attention_d24",
    "hat_attention_d30",
    "hat_unfold_kv",
    "hat_bias_gather",
    "hat_mask_build",
    "hat_window_index",
    "hat_plane_block",
    "hat_plane_edges",
    "hat_oca_label",
    "hat_token_range",
    "hat_plane_bias",
    "hat_pixel_shuffle",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/hat.cu");

    // `cargo build --no-default-features` is the pure-Rust CPU build: it must not
    // need nvcc, and `src/cuda.rs` (which includes the fatbins) is not compiled.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/hat.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/hat.cu").expect("read cuda/hat.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/hat.cu (it has {})",
            defined.join(", ")
        );
    }
    for d in &defined {
        assert!(
            PROJECT_KERNELS.contains(&d.as_str()),
            "cuda/hat.cu defines `{d}`, which PROJECT_KERNELS does not list -              it would be pruned from the fatbin and fail at launch"
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "hat_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/hat.cu",
            out_name: "hat_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}
