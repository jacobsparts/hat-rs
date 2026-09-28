//! PNG IO: decode to planar f32 in [0,1], encode RGB.
//!
//! HAT's reference test script reads with `cv2`, divides by 255, runs the network
//! and clamps to [0,1] before writing. This is that same float pipeline on the
//! host side of the device boundary, so nothing between the file and the kernels
//! does arithmetic of its own.
use std::fs::File;
use std::io::{BufWriter, Read, Write};

/// A planar RGB image: `data` is [3][h][w], contiguous, values in [0,1].
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(w: usize, h: usize) -> Image {
        Image { w, h, data: vec![0.0; 3 * w * h] }
    }

    #[inline]
    pub fn plane(&self, c: usize) -> &[f32] {
        let hw = self.w * self.h;
        &self.data[c * hw..(c + 1) * hw]
    }

    /// Interleaved RGB bytes, clamped and rounded the way the reference rounds.
    pub fn to_rgb8(&self) -> Vec<u8> {
        let hw = self.w * self.h;
        let mut out = vec![0u8; hw * 3];
        for i in 0..hw {
            for c in 0..3 {
                let v = self.data[c * hw + i].clamp(0.0, 1.0) * 255.0;
                out[i * 3 + c] = (v + 0.5).floor().min(255.0) as u8;
            }
        }
        out
    }
}

/// Decode a PNG (8/16-bit, grey/RGB/RGBA) into planar f32 RGB in [0,1].
pub fn load_rgb(path: &str) -> Result<Image, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
    load_rgb_stream(file).map_err(|e| format!("{}: {}", path, e))
}

/// The same decode from any reader, for `-i -`.
pub fn load_rgb_stream<R: Read>(src: R) -> Result<Image, String> {
    let mut reader = png::Decoder::new(src).read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => return Err(format!("unsupported png colour type {:?}", other)),
    };
    let sixteen = info.bit_depth == png::BitDepth::Sixteen;
    let sample_scale = match info.bit_depth {
        png::BitDepth::Eight => 1.0 / 255.0,
        png::BitDepth::Sixteen => 1.0 / 65535.0,
        other => return Err(format!("unsupported png bit depth {:?}", other)),
    };
    let bytes = &buf[..info.buffer_size()];
    let sample = |px: usize, c: usize| -> f32 {
        let idx = px * channels + c;
        if sixteen {
            let b0 = bytes[idx * 2] as u16;
            let b1 = bytes[idx * 2 + 1] as u16;
            ((b0 << 8) | b1) as f32 * sample_scale
        } else {
            bytes[idx] as f32 * sample_scale
        }
    };
    let mut img = Image::new(w, h);
    let hw = w * h;
    for y in 0..h {
        for x in 0..w {
            let px = y * w + x;
            let (r, g, b) = match channels {
                1 | 2 => {
                    let v = sample(px, 0);
                    (v, v, v)
                }
                _ => (sample(px, 0), sample(px, 1), sample(px, 2)),
            };
            img.data[px] = r;
            img.data[hw + px] = g;
            img.data[2 * hw + px] = b;
        }
    }
    Ok(img)
}

/// Write 8-bit RGB.
pub fn save_rgb(path: &str, w: usize, h: usize, rgb: &[u8]) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("create {}: {}", path, e))?;
    save_rgb_stream(BufWriter::new(file), w, h, rgb).map_err(|e| format!("{}: {}", path, e))
}

/// The same encode to any writer, for `-o -`.
pub fn save_rgb_stream<W: Write>(dst: W, w: usize, h: usize, rgb: &[u8]) -> Result<(), String> {
    assert_eq!(rgb.len(), w * h * 3);
    let mut enc = png::Encoder::new(dst, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgb).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(())
}
