# mpc-stemloop

Offline AI stem separation for short loops, running on the **Akai MPC Sample**.
Drop a loop on a pad, run a split, and get back drums / bass / other / vocals as
new samples -- entirely on the device, no internet. It is an offline "bounce this
loop to stems" utility, not a real-time insert effect. Original work; not derived
from or bundled with any other MPC mod.

## Install on your MPC Sample

**[Download the ready-to-flash SD card zip](https://github.com/pzuzarte/mpc-stemloop/releases/latest)**
(requires MPC Sample firmware **1.3.0.12**).

1. Unzip its contents onto the **top level (root)** of an SD card, so the card has
   an `autorun/` folder and `hdemucs_mmi_4s_net.onnx` at its root.
2. Insert the card into the MPC Sample and power on -- the mod loads automatically.
3. On the Sample page, select a loop (up to ~8 s), free 4 pads, and press
   **SHIFT + DATA** to split it into Drums / Bass / Other / Vocals on new pads.

The first split after power-on takes a little longer (it loads the model once);
later splits are quicker. To uninstall, delete the `autorun/` folder and
power-cycle -- nothing is written to the device's internal storage.

## How it works

The separation model is Hybrid Demucs (`hdemucs_mmi`). To keep it small and fast
on the device's CPU, mpc-stemloop uses a **host-STFT split**: the STFT and inverse
STFT run as plain code, and only the learned neural net runs in ONNX Runtime. That
keeps the Fourier math out of the model and keeps runtime memory bounded.

- The net runs as a fixed 4-second window. Shorter loops are zero-padded and
  trimmed back; longer loops (up to ~8 s) are processed in overlapping 4 s windows
  and cross-faded, so memory stays flat regardless of length.
- The engine (host STFT -> ONNX net -> inverse STFT -> 4 stems) is written in Rust
  and matches the Python reference to ~3e-7 (essentially bit-exact).
- On hardware it runs on the Rockchip RK3566 (quad Cortex-A55, 2 GB), loaded into
  the stock app via the SD-card autorun and `LD_PRELOAD`, using the open
  [`ioma8/mpc-sample-sdk`](https://github.com/ioma8/mpc-sample-sdk).

## Build from source

The model weights are not stored in git (they exceed GitHub's file-size limit);
they are regenerated from Demucs' public pretrained weights.

- **Models:** `pip install -r requirements.txt`, then export the net-only ONNX from
  `hdemucs_mmi` with `src/split_export.py`.
- **Try it on your computer:** `src/stemloop.py` separates a loop WAV into four stem
  WAVs with the same host-STFT pipeline.
- **The mod:** `rust/stemloop-mod` is a Rust `cdylib` built for aarch64 Linux
  (glibc), with the engine in `rust/engine` and the DSP in `rust/dsp`. Assemble the
  SD card as in the release zip: an `autorun/` folder (the built `.so`, the aarch64
  `libonnxruntime.so`, and the loader) plus `hdemucs_mmi_4s_net.onnx` at the root.

## Credits & license

Built on **Demucs** (Meta AI), the **mpc-sample-sdk** (ioma8), and **ONNX Runtime**
(Microsoft) -- all MIT-licensed. The `hdemucs_mmi` model weights are derived from
Demucs' pretrained release (MIT). This project's own code is MIT.

Not affiliated with, authorized by, or endorsed by inMusic / Akai Professional.
"MPC" and "MPC Sample" are trademarks of their respective owners. This is an
unofficial, experimental community modification -- use at your own risk.
