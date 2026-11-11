#!/usr/bin/env python3
"""ONNX-exportable STFT / iSTFT that matches demucs.spec numerically.

PyTorch's ONNX exporter cannot emit aten::istft (whole-model Demucs export
fails there) nor aten::unfold (framing). This expresses the STFT as a conv1d
and the iSTFT as a conv_transpose1d with windowed-DFT kernels -- which *is* the
STFT math, using only core ONNX ops (Conv, ConvTranspose, Pad).

Mirrors demucs.spec for pad=0:
    n_fft, hop=n_fft//4, hann(win_length=n_fft), normalized=True, center=True,
    pad_mode='reflect'. spectro returns stacked (real, imag), so no complex
    dtype ever enters the graph. ASCII-only output.
"""
import math
import torch
import torch.nn.functional as F


def _hann(n_fft, device, dtype):
    n = torch.arange(n_fft, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / n_fft)


def _angles(n_fft, device, dtype):
    n = torch.arange(n_fft, device=device, dtype=dtype)
    k = torch.arange(n_fft // 2 + 1, device=device, dtype=dtype)
    return 2.0 * math.pi * torch.outer(k, n) / n_fft    # [F, N]


def _stft_kernels(n_fft, device, dtype):
    ang = _angles(n_fft, device, dtype)                 # [F, N]
    w = _hann(n_fft, device, dtype)
    norm = 1.0 / math.sqrt(n_fft)                       # normalized=True
    ck = (torch.cos(ang) * w * norm).unsqueeze(1)       # [F, 1, N]
    sk = (-torch.sin(ang) * w * norm).unsqueeze(1)
    return ck, sk


def _istft_kernels(n_fft, device, dtype):
    ang = _angles(n_fft, device, dtype)                 # [F, N]
    w = _hann(n_fft, device, dtype)
    freqs = n_fft // 2 + 1
    scale = torch.full((freqs, 1), 2.0, device=device, dtype=dtype)
    scale[0, 0] = 1.0
    scale[-1, 0] = 1.0                                  # bins 0 and N/2 once
    denorm = math.sqrt(n_fft) / n_fft
    ick = (scale * torch.cos(ang) * w * denorm).unsqueeze(1)   # [F, 1, N]
    isk = (-scale * torch.sin(ang) * w * denorm).unsqueeze(1)
    wsq = (w * w).view(1, 1, n_fft)                     # [1, 1, N]
    return ick, isk, wsq


def spectro(x, n_fft=4096, hop_length=None, pad=0):
    assert pad == 0, "exportable path implements pad=0"
    hop = hop_length or n_fft // 4
    *other, length = x.shape
    x = x.reshape(-1, length).unsqueeze(1)              # [B, 1, L]
    dev, dt = x.device, x.dtype
    half = n_fft // 2
    xp = F.pad(x, (half, half), mode="reflect")         # center=True
    ck, sk = _stft_kernels(n_fft, dev, dt)
    zr = F.conv1d(xp, ck, stride=hop)                   # [B, F, T]
    zi = F.conv1d(xp, sk, stride=hop)
    F_ = n_fft // 2 + 1
    T = zr.shape[-1]
    return zr.reshape(*other, F_, T), zi.reshape(*other, F_, T)


def ispectro(zr, zi, hop_length=None, length=None, pad=0):
    assert pad == 0, "exportable path implements pad=0"
    *other, freqs, frames = zr.shape
    n_fft = 2 * freqs - 2
    hop = hop_length or n_fft // 4
    dev, dt = zr.device, zr.dtype
    zr = zr.reshape(-1, freqs, frames)
    zi = zi.reshape(-1, freqs, frames)
    ick, isk, wsq = _istft_kernels(n_fft, dev, dt)
    num = (F.conv_transpose1d(zr, ick, stride=hop)
           + F.conv_transpose1d(zi, isk, stride=hop))  # [B, 1, out]
    ones = torch.ones(zr.shape[0], 1, frames, device=dev, dtype=dt)
    env = F.conv_transpose1d(ones, wsq, stride=hop)     # [B, 1, out]
    out = (num / (env + 1e-8)).squeeze(1)               # [B, out]

    half = n_fft // 2
    if length is not None:
        target = length + 2 * half
        if out.shape[-1] < target:
            out = F.pad(out, (0, target - out.shape[-1]))
        out = out[:, half:-half][:, :length]
    else:
        out = out[:, half:-half]
    return out.reshape(*other, out.shape[-1])


# --- kernel-arg variants: kernels precomputed as static buffers so the ONNX
# --- exporter sees a known convolution kernel shape. ---
def spectro_k(x, ck, sk, hop):
    n_fft = ck.shape[-1]
    *other, length = x.shape
    x = x.reshape(-1, length).unsqueeze(1)
    half = n_fft // 2
    xp = F.pad(x, (half, half), mode="reflect")
    zr = F.conv1d(xp, ck, stride=hop)
    zi = F.conv1d(xp, sk, stride=hop)
    F_ = ck.shape[0]
    T = zr.shape[-1]
    return zr.reshape(*other, F_, T), zi.reshape(*other, F_, T)


def ispectro_k(zr, zi, ick, isk, wsq, hop, length):
    n_fft = ick.shape[-1]
    *other, freqs, frames = zr.shape
    zr = zr.reshape(-1, freqs, frames)
    zi = zi.reshape(-1, freqs, frames)
    num = (F.conv_transpose1d(zr, ick, stride=hop)
           + F.conv_transpose1d(zi, isk, stride=hop))
    ones = torch.ones(zr.shape[0], 1, frames, device=zr.device, dtype=zr.dtype)
    env = F.conv_transpose1d(ones, wsq, stride=hop)
    out = (num / (env + 1e-8)).squeeze(1)
    half = n_fft // 2
    target = length + 2 * half
    if out.shape[-1] < target:
        out = F.pad(out, (0, target - out.shape[-1]))
    out = out[:, half:-half][:, :length]
    return out.reshape(*other, out.shape[-1])
