//! The checkpoint: an mmap'd `.safetensors` plus the architecture it describes.
//!
//! THE FILE SAYS WHAT IT IS. HAT ships in three sizes whose tensors differ only in
//! their shapes - there is no name that appears in one and not another - so a
//! checkpoint loaded against the wrong configuration fails as a shape mismatch
//! deep inside the graph rather than at load. `tools/convert.py` therefore writes
//! the architecture into the container's `__metadata__` and this module reads it
//! back and checks it against the tensors that are actually present. A conversion
//! made for the wrong size, a stale conversion, or a hand-edited header all fail
//! here, where the message can say which.
use std::collections::BTreeMap;
use std::path::Path;

use lightgpu::safetensors;

/// The processed image the network sees: `(x - mean) * img_range`.
///
/// NOT ZERO, which matters: HAT subtracts the ImageNet-ish mean of the DIV2K
/// training set and adds it back at the end, so a save that skips the adjustment
/// produces a plausible-looking but wrong image rather than an obviously wrong one.
pub const RGB_MEAN: [f32; 3] = [0.4488, 0.4371, 0.4040];

/// One RHAG stage's copy of the OCAB's relative-position table is `(2*owin-1)^2`
/// rows by `heads`, where `owin = window_size + int(overlap_ratio * window_size)`.
/// The pair that appears in every released checkpoint is (961, 6) for the local
/// windows at `win = 16` and (1521, 6) for the OCAB's `owin = 24`, and both are
/// checked on load.
pub struct Weights {
    file: safetensors::File,
    /// The architecture name from the metadata, for the run header: "hat-s" etc.
    pub variant: String,
    pub scale: usize,
    pub window: usize,
    /// `window_size + int(overlap_ratio * window_size)`: the OCAB's window.
    pub owin: usize,
    pub embed: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: usize,
    /// Blocks per RHAG, one entry per stage.
    pub depths: Vec<usize>,
    /// `c // compress_ratio`: the width the CAB's two 3x3 convs squeeze to and
    /// expand from. Distinct from `squeezed`, which is the CHANNEL ATTENTION's
    /// hidden width inside the CAB (`c // squeeze_factor`) - for HAT-S both happen to
    /// be 6 (144//24), which is why the converted shapes alone cannot tell them apart.
    pub compressed: usize,
    /// `c // squeeze_factor`: the channel attention's hidden width.
    pub squeezed: usize,
    pub conv_scale: f32,
    pub img_range: f32,
    pub mean: [f32; 3],
    /// `num_feat` in the reconstruction head (64 for every released checkpoint).
    pub head_feat: usize,
    /// Every tensor name in the file, in the file's own order (for `--list-weights`).
    pub names: Vec<String>,
    pub bytes: u64,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Weights, String> {
        let path = path.as_ref();
        let file = safetensors::File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let mut meta: BTreeMap<String, String> = BTreeMap::new();
        for k in ["variant", "scale", "window_size", "overlap_ratio", "embed_dim", "num_heads",
                  "mlp_ratio", "depths", "compress_ratio", "squeeze_factor", "conv_scale",
                  "img_range", "rgb_mean", "source"] {
            let v = file
                .metadata_get(k)
                .ok_or_else(|| format!(
                    "{}: no `{k}` in __metadata__ - convert the checkpoint with tools/convert.py, \
                     which records the architecture; a plain state_dict cannot be checked",
                    path.display()))?;
            meta.insert(k.to_string(), v.to_string());
        }
        let num = |k: &str| -> Result<usize, String> {
            meta[k].parse().map_err(|e| format!("{}: `{}` = {:?}: {}", path.display(), k, meta[k], e))
        };
        let depths = meta["depths"]
            .split(',')
            .map(|s| s.trim().parse::<usize>().map_err(|e| format!("depths: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        let mean: Vec<f32> = meta["rgb_mean"]
            .split(',')
            .map(|s| s.trim().parse::<f32>().map_err(|e| format!("rgb_mean: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        if mean.len() != 3 {
            return Err(format!("rgb_mean has {} entries, expected 3", mean.len()));
        }
        let embed = num("embed_dim")?;
        let heads = num("num_heads")?;
        if heads == 0 || embed % heads != 0 {
            return Err(format!("embed_dim {embed} is not divisible by num_heads {heads}"));
        }
        let window = num("window_size")?;
        let overlap: f32 = meta["overlap_ratio"].parse().map_err(|e| format!("overlap_ratio: {e}"))?;
        // The reference's own expression, `int(window_size * overlap_ratio) + window_size`,
        // is an int() TRUNCATION towards zero - not a round - and the released value is
        // 0.5 at window 16, i.e. exactly 24. Writing it the same way keeps the two
        // in step if a checkpoint is ever trained with a different ratio.
        let owin = window + (window as f32 * overlap) as usize;

        let w = Weights {
            variant: meta["variant"].clone(),
            scale: num("scale")?,
            window,
            owin,
            embed,
            heads,
            head_dim: embed / heads,
            mlp_ratio: num("mlp_ratio")?,
            depths,
            compressed: embed / num("compress_ratio")?,
            squeezed: embed / num("squeeze_factor")?,
            conv_scale: meta["conv_scale"].parse().map_err(|e| format!("conv_scale: {e}"))?,
            img_range: meta["img_range"].parse().map_err(|e| format!("img_range: {e}"))?,
            mean: [mean[0], mean[1], mean[2]],
            // The head's width is not in the metadata because it is not a free
            // parameter of the released models; it is read from the tensor below so
            // that a checkpoint with a different one is not silently mis-run.
            head_feat: 0,
            names: file.order().to_vec(),
            bytes: file.len() as u64,
            file,
        };
        let mut w = w;
        w.head_feat = w.shape("conv_before_upsample.0.weight")[0];
        w.validate(path)?;
        Ok(w)
    }

    /// Check the file against the architecture it claims. These are the mistakes
    /// that otherwise surface as a wrong image rather than an error.
    fn validate(&self, path: &Path) -> Result<(), String> {
        let c = self.embed;
        let heads = self.heads;
        let mut want: Vec<(String, Vec<usize>)> = vec![
            ("conv_first.weight".into(), vec![c, 3, 3, 3]),
            ("conv_first.bias".into(), vec![c]),
            ("conv_after_body.weight".into(), vec![c, c, 3, 3]),
            ("conv_after_body.bias".into(), vec![c]),
            ("norm.weight".into(), vec![c]),
            ("norm.bias".into(), vec![c]),
            // The token LayerNorm the reference applies right after the flatten
            // (`patch_embed`'s own norm), which is present in every released
            // checkpoint - i.e. `patch_norm=True`.
            ("patch_embed.norm.weight".into(), vec![c]),
            ("patch_embed.norm.bias".into(), vec![c]),
            // The two relative-position index buffers, int64 in the checkpoint.
            ("relative_position_index_SA".into(), vec![self.window * self.window, self.window * self.window]),
            ("relative_position_index_OCA".into(), vec![self.window * self.window, self.owin * self.owin]),
        ];

        // The head. `num_feat` is 64 for every released checkpoint and is read from
        // the tensor rather than assumed, so a future one with a different width is
        // refused here rather than run with a mismatched buffer.
        want.push(("conv_before_upsample.0.weight".into(), vec![self.head_feat, c, 3, 3]));
        want.push(("conv_before_upsample.0.bias".into(), vec![self.head_feat]));
        let feat = self.head_feat;
        let blocks = self.up_blocks();
        if blocks.is_empty() {
            return Err(format!(
                "scale {} is not one the reference's own `Upsample` builds: it accepts a \
                 power of two (one {win}x{win}->4*{win} conv and a 2x shuffle per octave, so 2, \
                 4 and 8) and 3 (a single {win}x{win}->9*{win} conv and a 3x shuffle), and \
                 raises for anything else",
                self.scale,
                win = self.head_feat
            ));
        }
        // NUM_FEAT STAYS FIXED ACROSS THE BLOCKS. `Upsample` is a
        // `Sequential(Conv2d(num_feat, R*num_feat, 3, 1, 1), PixelShuffle(R))` per
        // block with the SAME num_feat carried through, so each block widens to
        // R*num_feat and the shuffle brings it back to num_feat at R times the
        // resolution. For a x4 model that is two octaves of [64 -> 256, shuffle] and
        // the checkpoint's `upsample.0.weight` and `upsample.2.weight` are BOTH
        // [256, 64, 3, 3] - a chain that quadrupled the width would expect
        // [256, 64] then [1024, 256] and be wrong on the second. The name index is
        // `2 * block` in both branches, because `Sequential` numbers the conv 0 and
        // the shuffle 1 whatever R is.
        for (o, (_r, widened)) in blocks.iter().enumerate() {
            want.push((format!("upsample.{}.weight", 2 * o), vec![*widened, feat, 3, 3]));
            want.push((format!("upsample.{}.bias", 2 * o), vec![*widened]));
        }
        want.push(("conv_last.weight".into(), vec![3, feat, 3, 3]));
        want.push(("conv_last.bias".into(), vec![3]));

        for (layer, depth) in self.depths.iter().enumerate() {
            want.push((format!("layers.{layer}.conv.weight"), vec![c, c, 3, 3]));
            want.push((format!("layers.{layer}.conv.bias"), vec![c]));
            for b in 0..*depth {
                let p = format!("layers.{layer}.residual_group.blocks.{b}");
                // `norm1`/`norm2` ARE REQUIRED, and the reason is worth recording
                // because an earlier version of this table omitted them. They are
                // `nn.LayerNorm` modules whose forward IS called, so their weight and
                // bias are trained: on the released HAT-S checkpoint block 0's
                // `norm1.weight` has mean 0.487 and std 0.141. Every key the model
                // needs is in `params_ema` - measured, 0 missing and 0 unexpected for
                // all 864 of them - so requiring these is what makes the engine load
                // the same network the reference evaluates.
                want.push((format!("{p}.norm1.weight"), vec![c]));
                want.push((format!("{p}.norm1.bias"), vec![c]));
                want.push((format!("{p}.norm2.weight"), vec![c]));
                want.push((format!("{p}.norm2.bias"), vec![c]));
                want.push((format!("{p}.attn.qkv.weight"), vec![3 * c, c]));
                want.push((format!("{p}.attn.qkv.bias"), vec![3 * c]));
                want.push((format!("{p}.attn.proj.weight"), vec![c, c]));
                want.push((format!("{p}.attn.proj.bias"), vec![c]));
                want.push((format!("{p}.attn.relative_position_bias_table"), vec![(2 * self.window - 1) * (2 * self.window - 1), heads]));
                want.push((format!("{p}.mlp.fc1.weight"), vec![self.mlp_ratio * c, c]));
                want.push((format!("{p}.mlp.fc1.bias"), vec![self.mlp_ratio * c]));
                want.push((format!("{p}.mlp.fc2.weight"), vec![c, self.mlp_ratio * c]));
                want.push((format!("{p}.mlp.fc2.bias"), vec![c]));
                // THE CAB'S CONVS ARE 3x3, not 1x1: the reference is
                // `Sequential(Conv2d(c, c//compress_ratio, 3, 1, 1), GELU,
                // Conv2d(c//compress_ratio, c, 3, 1, 1), ChannelAttention)`. The 1x1s
                // are the channel attention's, two lines down. This table originally
                // said 1x1 and the checkpoint disagreed.
                want.push((format!("{p}.conv_block.cab.0.weight"), vec![self.compressed, c, 3, 3]));
                want.push((format!("{p}.conv_block.cab.0.bias"), vec![self.compressed]));
                want.push((format!("{p}.conv_block.cab.2.weight"), vec![c, self.compressed, 3, 3]));
                want.push((format!("{p}.conv_block.cab.2.bias"), vec![c]));
                want.push((format!("{p}.conv_block.cab.3.attention.1.weight"), vec![self.squeezed, c, 1, 1]));
                want.push((format!("{p}.conv_block.cab.3.attention.1.bias"), vec![self.squeezed]));
                want.push((format!("{p}.conv_block.cab.3.attention.3.weight"), vec![c, self.squeezed, 1, 1]));
                want.push((format!("{p}.conv_block.cab.3.attention.3.bias"), vec![c]));
            }
            let p = format!("layers.{layer}.residual_group.overlap_attn");
            want.push((format!("{p}.norm1.weight"), vec![c]));
            want.push((format!("{p}.norm1.bias"), vec![c]));
            want.push((format!("{p}.norm2.weight"), vec![c]));
            want.push((format!("{p}.norm2.bias"), vec![c]));
            want.push((format!("{p}.qkv.weight"), vec![3 * c, c]));
            want.push((format!("{p}.qkv.bias"), vec![3 * c]));
            want.push((format!("{p}.proj.weight"), vec![c, c]));
            want.push((format!("{p}.proj.bias"), vec![c]));
            want.push((format!("{p}.relative_position_bias_table"), vec![(self.window + self.owin - 1) * (self.window + self.owin - 1), heads]));
            want.push((format!("{p}.mlp.fc1.weight"), vec![self.mlp_ratio * c, c]));
            want.push((format!("{p}.mlp.fc1.bias"), vec![self.mlp_ratio * c]));
            want.push((format!("{p}.mlp.fc2.weight"), vec![c, self.mlp_ratio * c]));
            want.push((format!("{p}.mlp.fc2.bias"), vec![c]));
        }
        for (name, shape) in &want {
            let got = self.file.shape(name).map_err(|_| {
                format!(
                    "{}: the checkpoint has no `{name}`; the metadata says {} with embed {} and \
                     depths {}, which does not describe these tensors",
                    path.display(),
                    self.variant,
                    c,
                    self.depths.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")
                )
            })?;
            if got != shape.as_slice() {
                return Err(format!(
                    "{}: `{name}` is {:?}, expected {:?} for {} at embed {}",
                    path.display(),
                    got,
                    shape,
                    self.variant,
                    c
                ));
            }
        }
        Ok(())
    }

    /// A weight tensor. Every name this engine asks for is validated at load, so a
    /// miss here is a programming error, not bad input.
    #[inline]
    pub fn t(&self, name: &str) -> &[f32] {
        self.file.f32(name).unwrap_or_else(|e| panic!("weight `{name}`: {e}"))
    }

    /// The reconstruction head's upsampling blocks, as `(factor, widened)` pairs.
    ///
    /// THE REFERENCE'S `Upsample` HAS TWO BRANCHES AND THEY ARE DIFFERENT SHAPES, not
    /// different counts: a power-of-two scale repeats `Conv2d(num_feat,
    /// 4*num_feat, 3)` + `PixelShuffle(2)` `log2(scale)` times, while scale 3 is a
    /// SINGLE `Conv2d(num_feat, 9*num_feat, 3)` + `PixelShuffle(3)`. Every site that
    /// needs the head's shape - the validator above, the CPU head, the GPU head and
    /// the footprint derivation - reads it HERE rather than taking a log2 of the
    /// scale, so the four cannot disagree about how many blocks there are or how wide
    /// each is. `w * 2` octaves and one `w * 3` block both end at the same output
    /// resolution for x6, which is why the factor has to be carried rather than
    /// derived from the count.
    ///
    /// An EMPTY result means the scale is neither 2^n (n >= 1) nor 3, which is
    /// exactly the set `Upsample.__init__` accepts - it raises otherwise. Callers
    /// turn that into an error rather than running a head that does not exist.
    pub fn up_blocks(&self) -> Vec<(usize, usize)> {
        let feat = self.head_feat;
        if self.scale == 3 {
            return vec![(3, 9 * feat)];
        }
        let mut s = self.scale;
        let mut n = 0;
        while s > 1 && s % 2 == 0 {
            n += 1;
            s /= 2;
        }
        if s != 1 {
            return Vec::new();
        }
        vec![(2, 4 * feat); n]
    }

    /// A tensor's shape, for the code that has to budget memory rather than read
    /// values (`memguard`). Every name asked about here was
    /// validated at load, so a miss here is a programming error.
    #[inline]
    pub fn shape(&self, name: &str) -> &[usize] {
        self.file.shape(name).unwrap_or_else(|e| panic!("weight `{name}`: {e}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.file.contains(name)
    }

    /// The int64 index buffers. They are not weights - the engine builds its own
    /// per-resolution index maps - but they are in every released checkpoint, and
    /// `--verify`/the tests use them to check that the engine's own construction
    /// agrees with the reference's for the window geometry.
    /// The int64 index buffers, reconstructed from the tensor's raw bytes.
    ///
    /// lightgpu's reader exposes `f32`/`to_f32`/`raw`/`info` and has NO int64
    /// accessor - a deliberate narrowness of the toolkit, since every operation it
    /// performs is fp32. The index buffers are the one int64 data in a checkpoint,
    /// so this crate owns the reinterpretation. The slice is checked against the
    /// tensor's own element count before it is built, so a truncated or mis-declared
    /// tensor is a load error rather than a short slice.
    pub fn i64(&self, name: &str) -> Vec<i64> {
        let info = self.file.info(name).unwrap_or_else(|e| panic!("index `{name}`: {e}"));
        assert_eq!(
            info.dtype,
            safetensors::DType::I64,
            "`{name}` is {:?}, not int64",
            info.dtype
        );
        let bytes = self.file.raw(name).unwrap_or_else(|e| panic!("index `{name}`: {e}"));
        let n = info.numel();
        assert_eq!(bytes.len(), n * 8, "`{name}`: {} bytes for {n} int64s", bytes.len());
        bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
            .collect()
    }
}
