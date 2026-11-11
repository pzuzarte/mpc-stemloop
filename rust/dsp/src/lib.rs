//! Host STFT/iSTFT for the MPC Sample stem-split pipeline.
//!
//! A Rust port of `src/host_dsp.py`, kept bit-parity with demucs
//! `HDemucs._spec` / `_ispec` (which wrap `torch.stft`/`torch.istft`). The split
//! pipeline runs the learned net in an inference runtime and does the transform
//! here, so no torch/demucs is needed at run time.
//!
//! Transform params (fixed by the model): `nfft = 4096`, `hop = 1024`, periodic
//! Hann, center (reflect pad `nfft/2`), normalized (divide by `sqrt(nfft)`).
//!
//! All transforms compute in `f64` to match numpy; public inputs/outputs are
//! `f32`. Complex spectra are laid out `(F, T)` row-major: index `f * t + t`.
//!
//! PCM/geometry conversions between `f32` samples, `f64` compute and `usize`
//! lengths are inherent to DSP code, so the two cast lints below are allowed for
//! the whole crate (the same choice minimod's `dsp` crate makes). Every other
//! cast is lossless and uses `f64::from`. Single-character names are the signal
//! conventions (`f` bin, `t` frame, `b`/`c`/`s` axes).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// FFT length used by the model.
pub const NFFT: usize = 4096;
/// Hop length used by the model.
pub const HOP: usize = 1024;

/// Periodic Hann window of length `n` (matches `torch.hann_window`, periodic).
fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect()
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// numpy `reflect` padding (reflect about the edge sample, no edge repeat).
/// Requires `left <= len-1` and `right <= len-1`.
fn reflect_pad(x: &[f64], left: usize, right: usize) -> Vec<f64> {
    let l = x.len();
    let mut out = Vec::with_capacity(left + l + right);
    for k in 0..left {
        out.push(x[left - k]);
    }
    out.extend_from_slice(x);
    for k in 0..right {
        out.push(x[l - 2 - k]);
    }
    out
}

/// Cached FFT plans plus the window, so repeated transforms do not re-plan.
pub struct Dsp {
    nfft: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    wsq: Vec<f64>,
    scale: f64,
}

impl Dsp {
    /// Build the plans and window for the model's `NFFT`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_nfft(NFFT)
    }

    /// Build for an explicit FFT length (tests / other models).
    #[must_use]
    pub fn with_nfft(nfft: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let window = hann_periodic(nfft);
        let wsq = window.iter().map(|w| w * w).collect();
        Self {
            nfft,
            forward: planner.plan_fft_forward(nfft),
            inverse: planner.plan_fft_inverse(nfft),
            window,
            wsq,
            scale: 1.0 / (nfft as f64).sqrt(),
        }
    }

    fn fbins(&self) -> usize {
        self.nfft / 2 + 1
    }

    /// Real FFT of one length-`nfft` frame: returns `nfft/2 + 1` bins,
    /// unnormalized (matches `numpy.fft.rfft`).
    fn rfft(&self, frame: &[f64]) -> Vec<Complex<f64>> {
        let mut buf: Vec<Complex<f64>> = frame.iter().map(|&r| Complex::new(r, 0.0)).collect();
        self.forward.process(&mut buf);
        buf.truncate(self.fbins());
        buf
    }

    /// Inverse real FFT: `nfft/2 + 1` Hermitian half -> `nfft` real samples,
    /// with `1/nfft` scaling (matches `numpy.fft.irfft`).
    fn irfft(&self, half: &[Complex<f64>]) -> Vec<f64> {
        let n = self.nfft;
        let mut full = vec![Complex::new(0.0, 0.0); n];
        full[..half.len()].copy_from_slice(half);
        for k in 1..n / 2 {
            full[n - k] = half[k].conj();
        }
        self.inverse.process(&mut full);
        let inv_n = 1.0 / n as f64;
        full.iter().map(|c| c.re * inv_n).collect()
    }

    /// STFT of one real row -> `(F, T)` complex, index `f * t_count + t`.
    /// Matches `host_dsp.stft_np` (center, normalized, reflect, periodic Hann).
    #[must_use]
    pub fn stft_row(&self, x: &[f64], hop: usize) -> (Vec<Complex<f64>>, usize, usize) {
        let n = self.nfft;
        let pad = n / 2;
        let xp = reflect_pad(x, pad, pad);
        let t_count = 1 + (xp.len() - n) / hop;
        let f = self.fbins();
        let mut out = vec![Complex::new(0.0, 0.0); f * t_count];
        let mut frame = vec![0.0_f64; n];
        for t in 0..t_count {
            let s = t * hop;
            for j in 0..n {
                frame[j] = xp[s + j] * self.window[j];
            }
            let half = self.rfft(&frame);
            for (fi, c) in half.iter().enumerate() {
                out[fi * t_count + t] = c * self.scale;
            }
        }
        (out, f, t_count)
    }

    /// iSTFT of one `(F, T)` spectrum -> real row of `length` samples.
    /// Matches `host_dsp.istft_np` (center, normalized).
    #[must_use]
    pub fn istft_row(
        &self,
        z: &[Complex<f64>],
        t_count: usize,
        hop: usize,
        length: usize,
    ) -> Vec<f64> {
        let n = self.nfft;
        let f = self.fbins();
        let sqrt_n = (n as f64).sqrt();
        let lp = n + hop * (t_count - 1);
        let mut out = vec![0.0_f64; lp];
        let mut wsum = vec![0.0_f64; lp];
        let mut half = vec![Complex::new(0.0, 0.0); f];
        for t in 0..t_count {
            for fi in 0..f {
                half[fi] = z[fi * t_count + t] * sqrt_n;
            }
            let frame = self.irfft(&half);
            let s = t * hop;
            for j in 0..n {
                out[s + j] += frame[j] * self.window[j];
                wsum[s + j] += self.wsq[j];
            }
        }
        for i in 0..lp {
            if wsum[i] > 1e-8 {
                out[i] /= wsum[i];
            }
        }
        let pad = n / 2;
        out[pad..pad + length].to_vec()
    }
}

impl Default for Dsp {
    fn default() -> Self {
        Self::new()
    }
}

/// `_spec`: mix `(b, c, l)` real -> `(zr, zi)` each flat `(b, c, 2048, le)`.
/// Returns `(zr, zi, le)`.
///
/// # Panics
/// Panics unless `dsp` was built for the model's `NFFT`.
#[must_use]
pub fn spec(dsp: &Dsp, mix: &[f32], b: usize, c: usize, l: usize) -> (Vec<f32>, Vec<f32>, usize) {
    assert_eq!(dsp.nfft, NFFT);
    let hl = HOP;
    let le = ceil_div(l, hl);
    let pad = hl / 2 * 3;
    let fr = 2048;
    let rows = b * c;
    let mut zr = vec![0.0_f32; rows * fr * le];
    let mut zi = vec![0.0_f32; rows * fr * le];
    for r in 0..rows {
        let row: Vec<f64> = mix[r * l..r * l + l]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let right = pad + le * hl - l;
        let xp = reflect_pad(&row, pad, right);
        let (z, _f, t_count) = dsp.stft_row(&xp, hl);
        // Z[:, :-1, :][:, :, 2:2+le]: drop bin 2048, take frames 2..2+le.
        for fi in 0..fr {
            for t2 in 0..le {
                let cval = z[fi * t_count + (t2 + 2)];
                let o = (r * fr + fi) * le + t2;
                zr[o] = cval.re as f32;
                zi[o] = cval.im as f32;
            }
        }
    }
    (zr, zi, le)
}

/// `_magnitude` (cac): `(b, c, 2048, le)` real/imag -> `(b, 2c, 2048, le)`,
/// interleaving each channel's real then imag plane.
#[must_use]
pub fn magnitude(zr: &[f32], zi: &[f32], b: usize, c: usize, le: usize) -> Vec<f32> {
    let fr = 2048;
    let c2 = c * 2;
    let mut mag = vec![0.0_f32; b * c2 * fr * le];
    for bi in 0..b {
        for ci in 0..c {
            let src = ((bi * c + ci) * fr) * le;
            let dst_r = ((bi * c2 + 2 * ci) * fr) * le;
            let dst_i = ((bi * c2 + 2 * ci + 1) * fr) * le;
            let n = fr * le;
            mag[dst_r..dst_r + n].copy_from_slice(&zr[src..src + n]);
            mag[dst_i..dst_i + n].copy_from_slice(&zi[src..src + n]);
        }
    }
    mag
}

/// `_mask` (cac reshape) + `_ispec`: net output `xspec (b, s, c2, 2048, le)` real
/// -> stems `(b, s, c, length)`, where `c = c2 / 2`.
///
/// The shape arguments mirror the net output tensor; they will be folded into a
/// shape struct when this crate is wired into the SDK mod.
///
/// # Panics
/// Panics unless `dsp` was built for the model's `NFFT` and `fr == 2048`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn mask_ispec(
    dsp: &Dsp,
    xspec: &[f32],
    b: usize,
    s: usize,
    c2: usize,
    fr: usize,
    le: usize,
    length: usize,
) -> Vec<f32> {
    assert_eq!(dsp.nfft, NFFT);
    assert_eq!(fr, 2048);
    let c = c2 / 2;
    let hl = HOP;
    let pad = hl / 2 * 3;
    let le_istft = hl * ceil_div(length, hl) + 2 * pad;
    let fr_p = fr + 1; // 2049
    let t_p = le + 4; // pad frames (2, 2)
    let mut stems = vec![0.0_f32; b * s * c * length];
    let mut z = vec![Complex::new(0.0, 0.0); fr_p * t_p];
    for bi in 0..b {
        for si in 0..s {
            for ci in 0..c {
                for v in &mut z {
                    *v = Complex::new(0.0, 0.0);
                }
                let base_r = (((bi * s + si) * c2 + 2 * ci) * fr) * le;
                let base_i = (((bi * s + si) * c2 + 2 * ci + 1) * fr) * le;
                for fi in 0..fr {
                    for t in 0..le {
                        let re = f64::from(xspec[base_r + fi * le + t]);
                        let im = f64::from(xspec[base_i + fi * le + t]);
                        // Fr pad (0,1): bin `fr` stays 0. T pad (2,2): offset +2.
                        z[fi * t_p + (t + 2)] = Complex::new(re, im);
                    }
                }
                let x = dsp.istft_row(&z, t_p, hl, le_istft);
                let dst = ((bi * s + si) * c + ci) * length;
                for i in 0..length {
                    stems[dst + i] = x[pad + i] as f32;
                }
            }
        }
    }
    stems
}
