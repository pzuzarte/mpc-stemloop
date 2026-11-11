//! The full stem-split pipeline, host side.
//!
//! loop audio -> deinterleave to stereo planar -> host STFT (`mpc_stemloop_dsp`)
//! -> magnitude -> ONNX net (`ort`) -> `_mask` + iSTFT -> add the time branch ->
//! four interleaved stem buffers (drums / bass / other / vocals).
//!
//! This mirrors the lean path of `src/stemloop.py`: the learned net runs in ONNX
//! Runtime, the transforms run here as exact real FFTs. The ORT session uses the
//! known-good config (ConstantFolding et al disabled), which is bit-exact vs the
//! Python reference.
//!
//! The net ONNX has fixed input shapes per whole-second bucket; one `Engine`
//! wraps one bucket's model.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::path::Path;

use mpc_stemloop_dsp as dsp;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

/// Sample rate the model is trained for.
pub const SR: usize = 44_100;
/// Number of demucs stems.
pub const STEMS: usize = 4;
const FR: usize = 2048; // spectral bins the net uses
const CH: usize = 2; // stereo
const C2: usize = CH * 2; // complex-as-channels

/// One separated stem: interleaved stereo f32, ready for the app's Extract.
pub struct Stem {
    /// Interleaved samples, length `frames * channels`.
    pub audio: Vec<f32>,
    /// Channel count (always 2 here).
    pub channels: u32,
    /// Frame count.
    pub frames: usize,
}

/// A loaded net bucket plus its ORT session.
pub struct Engine {
    session: Session,
    dsp: dsp::Dsp,
    /// Whole-second bucket length in samples (`secs * SR`).
    tgt: usize,
}

impl Engine {
    /// Load the net ONNX for a whole-second `secs` bucket and build the session.
    ///
    /// # Errors
    /// Returns an `ort` error if the environment, session or model fails to load.
    pub fn new(model_path: &Path, secs: usize) -> ort::Result<Self> {
        let env = ort::init().with_name("stemloop-engine").build()?;
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            // Phase-2 fix: these optimizers corrupt this graph in the ort path.
            .with_config_entry(
                "optimization.disable_specified_optimizers",
                "ConstantFolding;ConstantSharing;CommonSubexpressionElimination",
            )?
            // All 4 A55 cores for the net: a split is an offline bounce the user
            // waits on, so finishing sooner beats UI/audio responsiveness during
            // it (drop back to 3 if the device stutters badly while separating).
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        Ok(Self {
            session,
            dsp: dsp::Dsp::new(),
            tgt: secs * SR,
        })
    }

    /// Separate `input` (interleaved, `frames * channels` samples) into 4 stems.
    ///
    /// If `frames` fits the model window it runs in one pass; a longer loop is
    /// processed in overlapping windows (25% overlap) and crossfade-stitched, so
    /// peak memory stays at one window regardless of loop length.
    ///
    /// # Errors
    /// Returns an `ort` error if inference fails.
    ///
    /// # Panics
    /// Panics if `channels` is zero.
    pub fn separate(&mut self, input: &[f32], channels: usize, frames: usize) -> ort::Result<Vec<Stem>> {
        assert!(channels > 0, "channels must be > 0");
        let window = self.tgt;
        if frames <= window {
            return self.run_window(input, channels, frames);
        }
        // Overlapping windows + triangular-weighted overlap-add (as demucs does
        // for long audio): bounded memory, smooth seams.
        let overlap = window / 4;
        let stride = window - overlap;
        let mut out: Vec<Vec<f32>> = (0..STEMS).map(|_| vec![0.0_f32; frames * CH]).collect();
        let mut wsum = vec![0.0_f32; frames];
        let mut off = 0_usize;
        loop {
            let win_len = window.min(frames - off);
            let slice = &input[off * channels..(off + win_len) * channels];
            let chunk = self.run_window(slice, channels, win_len)?;
            for i in 0..win_len {
                let w = 1.0 + core::cmp::min(i, win_len - 1 - i) as f32;
                wsum[off + i] += w;
                for s in 0..STEMS {
                    let ca = &chunk[s].audio;
                    for c in 0..CH {
                        out[s][(off + i) * CH + c] += ca[i * CH + c] * w;
                    }
                }
            }
            if off + win_len >= frames {
                break;
            }
            off += stride;
        }
        for j in 0..frames {
            if wsum[j] > 0.0 {
                let inv = 1.0 / wsum[j];
                for stem in &mut out {
                    for c in 0..CH {
                        stem[j * CH + c] *= inv;
                    }
                }
            }
        }
        Ok(out
            .into_iter()
            .map(|audio| Stem {
                channels: CH as u32,
                frames,
                audio,
            })
            .collect())
    }

    /// Run the net on one window of `win_frames` (<= the model window),
    /// zero-padded to the window; mono is upmixed to dual-mono.
    fn run_window(
        &mut self,
        input: &[f32],
        channels: usize,
        win_frames: usize,
    ) -> ort::Result<Vec<Stem>> {
        let tgt = self.tgt;
        let out_len = win_frames.min(tgt);

        // Deinterleave to stereo planar (2, tgt), zero-padded; mono -> dual mono.
        let mut mix = vec![0.0_f32; CH * tgt];
        for i in 0..out_len {
            for c in 0..CH {
                let src_c = if channels == 1 { 0 } else { c.min(channels - 1) };
                mix[c * tgt + i] = input[i * channels + src_c];
            }
        }

        // Host STFT -> cac magnitude (the net's "mag" input).
        let (zr, zi, le) = dsp::spec(&self.dsp, &mix, 1, CH, tgt);
        let mag = dsp::magnitude(&zr, &zi, 1, CH, le);

        // ONNX net: (mag, mix) -> (xspec, xt).
        let mag_t = Tensor::from_array(([1_usize, C2, FR, le], mag))?;
        let mix_t = Tensor::from_array(([1_usize, CH, tgt], mix))?;
        let outputs = self
            .session
            .run(ort::inputs!["mag" => mag_t, "mix" => mix_t])?;
        let (_, xspec) = outputs["xspec"].try_extract_tensor::<f32>()?;
        let xspec = xspec.to_vec();
        let (_, xt) = outputs["xt"].try_extract_tensor::<f32>()?;
        let xt = xt.to_vec();

        // _mask + iSTFT for the spectral branch, then add the time branch.
        let stems_spec = dsp::mask_ispec(&self.dsp, &xspec, 1, STEMS, C2, FR, le, out_len);

        let mut out = Vec::with_capacity(STEMS);
        for s in 0..STEMS {
            let mut audio = vec![0.0_f32; out_len * CH];
            for i in 0..out_len {
                for c in 0..CH {
                    let spec_v = stems_spec[(s * CH + c) * out_len + i];
                    let xt_v = xt[(s * CH + c) * tgt + i];
                    audio[i * CH + c] = spec_v + xt_v;
                }
            }
            out.push(Stem {
                audio,
                channels: CH as u32,
                frames: out_len,
            });
        }
        Ok(out)
    }
}
