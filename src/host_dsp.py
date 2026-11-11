#!/usr/bin/env python3
"""Pure-numpy host STFT/iSTFT for the StemLoop split pipeline.

Replicates demucs HDemucs._spec / _ispec (which wrap torch.stft/istft) exactly,
using numpy.fft only -- so the inference runtime is ONNX Runtime + numpy, with no
torch / demucs loaded (that framework is ~450 MB of RSS we don't want on device).
This is the host side the on-device C FFT will mirror.

Transform params (fixed by the model): nfft=4096, hop=1024, periodic Hann,
center=True (reflect pad nfft//2), normalized=True (divide by sqrt(nfft)).
The demucs-specific extra padding / frame-trim is reproduced in spec_np/ispec_np.

ASCII-only.
"""
import numpy as np

NFFT = 4096
HOP = 1024


def hann_periodic(n):
    # torch.hann_window default is periodic (denominator n, not n-1).
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


_W = hann_periodic(NFFT)
_WSQ_WIN = _W * _W


def _reflect_pad(x, left, right):
    # numpy reflect == torch 'reflect' (no edge repeat). x: (..., L)
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode="reflect")


def stft_np(x, n_fft=NFFT, hop=HOP):
    """x: (B, L) real -> (B, F=n_fft/2+1, T) complex. Matches torch.stft(center,
    normalized, reflect, periodic-hann)."""
    pad = n_fft // 2
    xp = _reflect_pad(x, pad, pad)
    L = xp.shape[-1]
    T = 1 + (L - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(T)[:, None]   # (T, n_fft)
    frames = xp[:, idx] * _W[None, None, :]                         # (B, T, n_fft)
    Z = np.fft.rfft(frames, n=n_fft, axis=-1) / np.sqrt(n_fft)      # (B, T, F)
    return np.transpose(Z, (0, 2, 1))                              # (B, F, T)


def istft_np(Z, n_fft=NFFT, hop=HOP, length=None):
    """Z: (B, F, T) complex -> (B, length) real. Matches torch.istft(center,
    normalized)."""
    B, F, T = Z.shape
    Zs = np.transpose(Z, (0, 2, 1)) * np.sqrt(n_fft)               # (B, T, F), undo norm
    frames = np.fft.irfft(Zs, n=n_fft, axis=-1) * _W[None, None, :]  # (B, T, n_fft)
    Lp = n_fft + hop * (T - 1)
    out = np.zeros((B, Lp), dtype=np.float64)
    wsum = np.zeros(Lp, dtype=np.float64)
    for t in range(T):
        s = t * hop
        out[:, s:s + n_fft] += frames[:, t, :]
        wsum[s:s + n_fft] += _WSQ_WIN
    nz = wsum > 1e-8
    out[:, nz] /= wsum[nz]
    pad = n_fft // 2
    if length is None:
        length = Lp - 2 * pad
    return out[:, pad:pad + length]


def _ceil_div(a, b):
    return -(-a // b)


def spec_np(mix):
    """mix: (1, 2, L) real -> (zr, zi) each (1, 2, 2048, le). Mirrors _spec."""
    B, C, L = mix.shape
    hl = HOP
    le = _ceil_div(L, hl)
    pad = hl // 2 * 3
    x = _reflect_pad(mix.reshape(B * C, L), pad, pad + le * hl - L)   # (B*C, Lp)
    Z = stft_np(x, NFFT, hl)                                          # (B*C, 2049, frames)
    Z = Z[:, :-1, :][:, :, 2:2 + le]                                 # (B*C, 2048, le)
    Z = Z.reshape(B, C, 2048, le)
    return np.real(Z).astype(np.float32), np.imag(Z).astype(np.float32)


def magnitude_np(zr, zi):
    """(1,2,2048,le) real/imag -> mag (1, 4, 2048, le) [cac: real/imag -> channels]."""
    B, C, Fr, T = zr.shape
    m = np.stack([zr, zi], axis=2)                 # (B, C, 2, Fr, T)
    return m.reshape(B, C * 2, Fr, T).astype(np.float32)


def mask_ispec_np(xspec, length):
    """xspec: net output (1, S, C2, 2048, le) real -> stems (1, S, 2, length).
    Mirrors _mask (cac reshape) + _ispec (pad + istft + trim)."""
    B, S, _, Fr, T = xspec.shape
    m = xspec.reshape(B, S, -1, 2, Fr, T)          # (B,S,C,2,Fr,T)
    zr = m[:, :, :, 0]                              # (B,S,C,Fr,T)
    zi = m[:, :, :, 1]
    C = zr.shape[2]
    hl = HOP
    # _ispec: pad Fr by (0,1) and T by (2,2)
    zr = np.pad(zr, [(0, 0), (0, 0), (0, 0), (0, 1), (2, 2)])
    zi = np.pad(zi, [(0, 0), (0, 0), (0, 0), (0, 1), (2, 2)])
    pad = hl // 2 * 3
    le = hl * _ceil_div(length, hl) + 2 * pad
    Z = (zr + 1j * zi).reshape(B * S * C, zr.shape[3], zr.shape[4])
    x = istft_np(Z, NFFT, hl, length=le)           # (B*S*C, le)
    x = x[:, pad:pad + length]
    return x.reshape(B, S, C, length).astype(np.float32)
