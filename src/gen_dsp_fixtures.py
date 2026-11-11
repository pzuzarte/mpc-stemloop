#!/usr/bin/env python3
"""Generate parity fixtures for the Rust DSP port.

Runs deterministic inputs through host_dsp.py and writes every input and expected
output as raw little-endian float32, plus a manifest.json describing each array.
The Rust tests (rust/dsp/tests/parity.rs) read the SAME input bytes, run the Rust
transforms, and compare to the expected arrays.

Inputs are generated as float32 and cast up to float64 for the numpy compute, so
both sides start from identical bits (no float32-quantization mismatch).

ASCII-only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host_dsp as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "rust", "dsp", "tests", "fixtures")
OUT = os.path.abspath(OUT)

SEED = 1234
L_SIG = 7000       # single-channel length for stft/istft
L_MIX = 7000       # mix length for spec/magnitude/mask_ispec


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4")


manifest = {}


def dump(name, arr):
    a = f32(arr)
    a.tofile(os.path.join(OUT, name + ".bin"))
    manifest[name] = {"shape": list(a.shape), "dtype": "f32", "path": name + ".bin"}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- stft: x (1, L) real -> Zr, Zi (1, 2049, T) ---
    x = f32(rng.standard_normal((1, L_SIG)) * 0.3)
    Z = H.stft_np(x.astype(np.float64))
    dump("stft_in_x", x)
    dump("stft_out_zr", np.real(Z))
    dump("stft_out_zi", np.imag(Z))

    # --- istft: Z (1, 2049, T) complex -> out (1, length) real ---
    # reuse the stft output as the istft input; fixed explicit length.
    zr_in = f32(np.real(Z))
    zi_in = f32(np.imag(Z))
    Zin = zr_in.astype(np.float64) + 1j * zi_in.astype(np.float64)
    out = H.istft_np(Zin, length=L_SIG)
    dump("istft_in_zr", zr_in)
    dump("istft_in_zi", zi_in)
    dump("istft_out", out)

    # --- spec: mix (1, 2, L) -> zr, zi (1, 2, 2048, le) ---
    mix = f32(rng.standard_normal((1, 2, L_MIX)) * 0.3)
    zr, zi = H.spec_np(mix.astype(np.float64))
    dump("spec_in_mix", mix)
    dump("spec_out_zr", zr)
    dump("spec_out_zi", zi)

    # --- magnitude: (zr, zi) -> mag (1, 4, 2048, le) ---
    mag = H.magnitude_np(zr, zi)
    dump("mag_out", mag)

    # --- mask_ispec: xspec (1, S, C2, 2048, le) -> stems (1, S, C, length) ---
    S, C2, Fr = 4, 4, 2048
    le = zr.shape[-1]
    xspec = f32(rng.standard_normal((1, S, C2, Fr, le)) * 0.3)
    stems = H.mask_ispec_np(xspec.astype(np.float64), L_MIX)
    dump("maskispec_in_xspec", xspec)
    dump("maskispec_out", stems)

    manifest["_params"] = {"nfft": H.NFFT, "hop": H.HOP,
                           "l_sig": L_SIG, "l_mix": L_MIX}
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("wrote fixtures to", OUT)
    for k, v in manifest.items():
        if k != "_params":
            print("  %-20s %s" % (k, v["shape"]))


if __name__ == "__main__":
    main()
