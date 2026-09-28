HAT architecture as pinned by hat_arch.py + HAT-S_SRx4.pth (params_ema)

forward: x = (x - mean) * img_range            mean=(0.4488,0.4371,0.4040) for 3ch, img_range=1.0
         x = conv_first(x)                     3 -> embed, 3x3, pad 1
         f = forward_features(x)
         x = conv_after_body(f) + x            embed -> embed 3x3
         x = conv_before_upsample(x)           {Conv2d(embed,64,3,1,1), LeakyReLU(0.01)}
         x = conv_last(upsample(x))            {Conv2d(64,256,3,1,1), PixelShuffle(2)} * log2(scale); Conv2d(64,3,3,1,1)
         out = x / img_range + mean
forward_features: t = patch_embed(x)          flatten(2).transpose(1,2) then LayerNorm(embed)   [patch_embed.norm exists]
         for layer in layers: t = RHAG(t, (h,w), params)
         t = norm(t)                          final LayerNorm(embed)
         return patch_unembed(t, x_size)       transpose -> [b, c, h, w]

RHAG(x): r = residual_group(unembed(x_view))  x is [b, hw, c]; patch_unembed -> [b,c,h,w]
         return patch_embed(conv3x3(r)) + x    resi_connection '1conv': Conv2d(embed,embed,3,1,1)
residual_group = AttenBlocks: for blk in blocks[HAB]*depth: x = blk(x, x_size, {rpi_sa, attn_mask})
                              then x = overlap_attn(x, x_size, rpi_oca)

HAB(x):  shortcut = x; xn = norm1(x); xr = xn.view(b,h,w,c)
         conv_x = CAB(xr.permute(0,3,1,2)).permute(0,2,3,1).reshape(b,hw,c)     (NCHW 3x3 path!)
         shifted = roll(xr, (-s,-s), dims=(1,2)) if s>0 else xr
         w = window_partition(shifted, 16).reshape(-1,256,c)
         a = WindowAttention(w, rpi_sa, mask if s>0 else None)
         a = window_reverse(a.view(-1,16,16,c)); unshift if s>0 (roll (+s,+s)); a = a.view(b,hw,c)
         x = shortcut + a + conv_x*0.01
         x = x + mlp(norm2(x))                 fc1 embed->2*embed (GELU), fc2 2*embed->embed
CAB(x):  y = x + conv3x3_1x1(x)               Sequential(c, c//compress, 1) ; ReLU ; (c//compress, c, 1)
         y = y + ChannelAttention(y)          AdaptiveAvgPool2d(1) over HW -> (c,c//squeeze,1) ReLU (c//squeeze,c,1) Sigmoid
ChannelAttention multiplies the pooled channel vector back over H,W.

WindowAttention(w[b_,256,c], rpi[256,256], mask[nw,256,256]):
    qkv = Linear(c, 3c); reshape (b_,256,3,nH,d); q,k,v = qkv[i] -> [b_, nH, 256, d]
    q *= d**-0.5 ; attn = q @ k^T                                  [b_, nH, 256, 256]
    bias = table[rpi.view(-1)].view(256,256,nH).permute(2,0,1)      [nH,256,256]
    attn += bias ; if mask: attn = attn.view(b_//nw, nw, nH, 256, 256) + mask[:,None]
    attn = softmax(attn, -1) ; x = (attn @ v).transpose(1,2).reshape(b_,256,c) ; proj(c,c)
    NOTE b_ = nw*b with the WINDOW index slower than the batch index (b_//nw, nw, ...).

OCAB(x): shortcut = x; xn = norm1(x); xr = [b,h,w,c]
         qkv = Linear(c,3c)(xr).reshape(b,h,w,3,c).permute(3,0,4,1,2)   -> [3,b,c,h,w]
         q = qkv[0].permute(0,2,3,1)                                    [b,h,w,c]
         kv = cat((qkv[1], qkv[2]), dim=1)                              [b, 2c, h, w]
         q_win = window_partition(q,16).reshape(-1,256,c)
         kv_win = Unfold(kernel=13, stride=16, padding=3)(kv) -> [b, 2c*169, nw]
                  rearrange 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch'
                  => Unfold's channel order is C-major (nc outer, ch inner); zero padding at the borders
         k,v = [nw*b, 169, c]
         attn = q @ k^T * scale                                          [b_, nH, 256, 169]
         bias = table[rpi_oca.view(-1)].view(256,169,nH).permute(2,0,1)   [nH,256,169]
         attn = softmax(attn + bias, -1)                                  (no mask)
         a = (attn @ v).transpose(1,2).reshape(b_,256,c) -> window_reverse -> [b,hw,c]
         x = proj(a) + shortcut ; x = x + mlp(norm2(x))
  overlap_win_size = int(16*0.5)+16 = 24 ; rpi_oca built from window_size_ext=24 coords vs 16 coords:
  relative_coords = coords_ext[:,None,:] - coords_ori[:,:,None]  -> [256,169,2] (WATCH the asymmetry)
  shift = ws_ori - ws_ext + 1 = -7 ; index = (dh-7)*(16+24-1) + (dw-7)

rpi_sa: coords = meshgrid(16,16) flatten -> [2,256]; relative = c[:,:,None]-c[:,None,:] permute(1,2,0)
        +15 both ; dh *= 31 ; index = dh*31 + dw  (so index = (dh+15)*31 + (dw+15))
table shapes: SA [961,6] = 31*31 ; OCA [1521,6] = 39*39

attn_mask = calculate_mask(h,w) for the SHIFTED blocks only (odd block index), shift = 8:
  img_mask slices per axis: (0:-16), (-16:-8), (-8:) -> 3x3 = 9 regions numbered row-major from 0
  mask = (mw[:,None]-mw[None,:]) != 0 -> -100.0 else 0.0, per window, shape [nw,256,256]

Checkpoint evidence: patch_norm True (patch_embed.norm present), ape False, norm (final) present,
no PatchMerging tensors (downsample None). qkv [432,144], proj [144,144], fc1 [288,144], fc2 [144,288].
