#!/usr/bin/env python3
"""Patch an HTDemucs instance so no complex tensor enters the graph.

htdemucs runs with cac=True (complex-as-channels): the complex spectrogram is
only ever a container for two real channels, never used in complex arithmetic
(no abs, no Wiener). So we represent it as a real tensor with a trailing dim of
size 2 (real, imag) and swap in our ONNX-exportable spectro/ispectro.

patch(model) rebinds _spec, _magnitude, _mask, _ispec on the instance. The math
is identical to stock htdemucs; only the complex dtype is removed. ASCII only.
"""
import math
import types
import torch
import torch.nn.functional as F
from demucs.htdemucs import pad1d
from demucs.demucs import LocalState
import onnx_spec


def _localstate_forward(self, x):
    """Identical to demucs LocalState.forward, except the diagonal self-mask
    uses `delta == 0` instead of torch.eye(T). torch.eye exports to the ONNX
    EyeLike op, which ONNX Runtime's CPU provider does not implement; the
    equality on the already-computed `delta` exports to Equal and is numerically
    identical (both mark exactly the t==s diagonal).
    """
    B, C, T = x.shape
    heads = self.heads
    indexes = torch.arange(T, device=x.device, dtype=x.dtype)
    delta = indexes[:, None] - indexes[None, :]

    queries = self.query(x).view(B, heads, -1, T)
    keys = self.key(x).view(B, heads, -1, T)
    dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
    dots /= keys.shape[2] ** 0.5
    if self.nfreqs:
        periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
        freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
        freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs ** 0.5
        dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
    if self.ndecay:
        decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
        decay_q = self.query_decay(x).view(B, heads, -1, T)
        decay_q = torch.sigmoid(decay_q) / 2
        decay_kernel = - decays.view(-1, 1, 1) * delta.abs() / self.ndecay ** 0.5
        dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

    # Kill self reference (diagonal) without torch.eye / EyeLike.
    dots = dots.masked_fill(delta == 0, -100)
    weights = torch.softmax(dots, dim=2)

    content = self.content(x).view(B, heads, -1, T)
    result = torch.einsum("bhts,bhct->bhcs", weights, content)
    if self.nfreqs:
        time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
        result = torch.cat([result, time_sig], 2)
    result = result.reshape(B, -1, T)
    return x + self.proj(result)


def _spec_ri(self, x):
    hl = self.hop_length
    nfft = self.nfft
    assert hl == nfft // 4
    le = int(math.ceil(x.shape[-1] / hl))
    pad = hl // 2 * 3
    x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
    zr, zi = onnx_spec.spectro_k(x, self._osp_ck, self._osp_sk, hl)
    zr = zr[..., :-1, :][..., 2:2 + le]
    zi = zi[..., :-1, :][..., 2:2 + le]
    return torch.stack([zr, zi], dim=-1)             # [B, C, Fr, T, 2]


def _magnitude_ri(self, z):
    # cac path: move the real/imag dim into channels.
    B, C, Fr, T, _ = z.shape
    return z.permute(0, 1, 4, 2, 3).reshape(B, C * 2, Fr, T)


def _mask_ri(self, z, m):
    # cac path: m is a full spectrogram; return it as [B, S, C, Fr, T, 2] real.
    B, S, C, Fr, T = m.shape
    out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
    return out.contiguous()


def _ispec_ri(self, z, length=None, scale=0):
    hl = self.hop_length // (4 ** scale)
    zr = z[..., 0]
    zi = z[..., 1]
    # Match stock: F.pad(z,(0,0,0,1)) -> pad Fr by (0,1); F.pad(z,(2,2)) -> T by (2,2).
    zr = F.pad(zr, (0, 0, 0, 1)); zi = F.pad(zi, (0, 0, 0, 1))
    zr = F.pad(zr, (2, 2)); zi = F.pad(zi, (2, 2))
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    x = onnx_spec.ispectro_k(zr, zi, self._osp_ick, self._osp_isk,
                             self._osp_wsq, hl, le)
    return x[..., pad: pad + length]


def fix_localstate(model):
    """Replace torch.eye self-mask in every LocalState (DConv attention) with
    `delta == 0`, so the graph has no EyeLike op (ORT CPU lacks it). Needed on
    its own for the host-STFT split, where the DFT stays off the graph and the
    other spectral rebinds are not wanted. hdemucs_mmi has LocalState; htdemucs
    may not. Returns the count patched."""
    n_ls = 0
    for mod in model.modules():
        if isinstance(mod, LocalState):
            mod.forward = types.MethodType(_localstate_forward, mod)
            n_ls += 1
    model._osp_n_localstate = n_ls
    return n_ls


def patch(model):
    """Rebind the spectral methods on a (single) HTDemucs instance.

    Precomputes STFT/iSTFT kernels as static buffers (known shape) so the ONNX
    exporter accepts the conv / conv_transpose kernels. This is the WHOLE-MODEL
    export path (STFT/iSTFT baked into the graph). The host-STFT split uses
    fix_localstate() only and keeps stock real-FFT _spec/_ispec on the host.
    """
    assert getattr(model, "cac", False), "patch supports cac=True models (htdemucs)"
    n_fft = model.nfft
    p = next(model.parameters())
    dev, dt = p.device, p.dtype
    ck, sk = onnx_spec._stft_kernels(n_fft, dev, dt)
    ick, isk, wsq = onnx_spec._istft_kernels(n_fft, dev, dt)
    model.register_buffer("_osp_ck", ck, persistent=False)
    model.register_buffer("_osp_sk", sk, persistent=False)
    model.register_buffer("_osp_ick", ick, persistent=False)
    model.register_buffer("_osp_isk", isk, persistent=False)
    model.register_buffer("_osp_wsq", wsq, persistent=False)
    model._spec = types.MethodType(_spec_ri, model)
    model._magnitude = types.MethodType(_magnitude_ri, model)
    model._mask = types.MethodType(_mask_ri, model)
    model._ispec = types.MethodType(_ispec_ri, model)
    # Replace torch.eye self-mask in every LocalState (DConv attention) so the
    # graph has no EyeLike op. hdemucs_mmi has these; htdemucs may not.
    n_ls = 0
    for mod in model.modules():
        if isinstance(mod, LocalState):
            mod.forward = types.MethodType(_localstate_forward, mod)
            n_ls += 1
    model._osp_n_localstate = n_ls
    return model
