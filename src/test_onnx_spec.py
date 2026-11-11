#!/usr/bin/env python3
"""Verify onnx_spec.spectro/ispectro match demucs.spec within tolerance.

Checks, for htdemucs params (n_fft=4096, hop=1024, pad=0):
  1) spectro real/imag vs demucs complex stft
  2) ispectro vs demucs istft (on the demucs spectrogram)
  3) round-trip ispectro(spectro(x)) vs x
ASCII-only output.
"""
import torch
import demucs.spec as dspec
import onnx_spec as o

torch.manual_seed(0)
N_FFT = 4096
HOP = N_FFT // 4
L = 4 * 44100 // 10  # ~0.4s is enough to exercise many frames


def main():
    x = torch.randn(2, L, dtype=torch.float32)

    # 1) forward
    z = dspec.spectro(x, n_fft=N_FFT, hop_length=HOP, pad=0)   # complex [.,F,T]
    zr, zi = o.spectro(x, n_fft=N_FFT, hop_length=HOP, pad=0)
    er = (zr - z.real).abs().max().item()
    ei = (zi - z.imag).abs().max().item()
    print("spectro  max err real=%.2e imag=%.2e" % (er, ei))

    # 2) inverse on the SAME demucs spectrogram
    x_ref = dspec.ispectro(z, hop_length=HOP, length=L, pad=0)
    x_mine = o.ispectro(z.real, z.imag, hop_length=HOP, length=L, pad=0)
    ei2 = (x_mine - x_ref).abs().max().item()
    print("ispectro max err vs demucs istft = %.2e" % ei2)

    # 3) round trip through my own pair
    zr2, zi2 = o.spectro(x, n_fft=N_FFT, hop_length=HOP, pad=0)
    xr = o.ispectro(zr2, zi2, hop_length=HOP, length=L, pad=0)
    rt = (xr - x).abs().max().item()
    print("round-trip max err ispectro(spectro(x)) vs x = %.2e" % rt)

    ok = er < 1e-3 and ei < 1e-3 and ei2 < 1e-3 and rt < 1e-3
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
