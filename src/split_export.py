#!/usr/bin/env python3
"""Host-side STFT split: export ONLY the learned net, keep STFT/iSTFT on the host.

HDemucs is hybrid: a spectral branch (on the STFT) and a time branch (on the raw
waveform) that merge inside the net. But the net touches the Fourier transform in
exactly two weightless places:
  * forward entry : z = _spec(mix); mag = _magnitude(z)   (STFT, then a reshape)
  * forward exit  : zout = _mask(z, x); x = _ispec(zout)  (reshape, then iSTFT)
For cac=True models _magnitude and _mask are pure reshapes and _mask ignores z.

So we cut the net at those two points. The exported graph takes (mag, mix) and
returns (xspec_mask, xt): the denormalized spectral output (cac layout) and the
denormalized time-branch waveform. The HOST computes, in fp32:
    z = spec(mix); mag = magnitude(z)          # before
    xspec_mask, xt = net(mag, mix)             # ONNX
    xspec = ispec(mask(xspec_mask), length)    # after
    stems = xt + xspec

Win: the ~135 MB of DFT conv kernels leave every exported model, and the fp16/
INT8-hostile DFT math stays in exact fp32 host code -- so the net alone is what
gets quantized for the NPU. ASCII-only output.
"""
import types
import torch
import htdemucs_onnx


def net_only_forward(self, mag, mix):
    """HDemucs.forward with _spec/_magnitude removed from the front and
    _mask/_ispec/add removed from the back. Body copied verbatim from
    demucs.hdemucs.HDemucs.forward (the encoder/decoder loop is unchanged)."""
    x = mag
    B, C, Fq, T = x.shape
    length = mix.shape[-1]

    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (1e-5 + std)

    if self.hybrid:
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

    saved = []
    saved_t = []
    lengths = []
    lengths_t = []
    for idx, encode in enumerate(self.encoder):
        lengths.append(x.shape[-1])
        inject = None
        if self.hybrid and idx < len(self.tencoder):
            lengths_t.append(xt.shape[-1])
            tenc = self.tencoder[idx]
            xt = tenc(xt)
            if not tenc.empty:
                saved_t.append(xt)
            else:
                inject = xt
        x = encode(x, inject)
        if idx == 0 and self.freq_emb is not None:
            frs = torch.arange(x.shape[-2], device=x.device)
            emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
            x = x + self.freq_emb_scale * emb
        saved.append(x)

    x = torch.zeros_like(x)
    if self.hybrid:
        xt = torch.zeros_like(x)

    for idx, decode in enumerate(self.decoder):
        skip = saved.pop(-1)
        x, pre = decode(x, skip, lengths.pop(-1))
        if self.hybrid:
            offset = self.depth - len(self.tdecoder)
        if self.hybrid and idx >= offset:
            tdec = self.tdecoder[idx - offset]
            length_t = lengths_t.pop(-1)
            if tdec.empty:
                assert pre.shape[2] == 1, pre.shape
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, length_t)
            else:
                skip = saved_t.pop(-1)
                xt, _ = tdec(xt, skip, length_t)

    assert len(saved) == 0
    assert len(lengths_t) == 0
    assert len(saved_t) == 0

    S = len(self.sources)
    x = x.view(B, S, -1, Fq, T)
    x = x * std[:, None] + mean[:, None]

    xt = xt.view(B, S, -1, length)
    xt = xt * stdt[:, None] + meant[:, None]
    return x, xt


def prepare(model_name):
    """Return a net whose forward is (mag, mix) -> (xspec_mask, xt).

    Only the LocalState EyeLike fix is applied; the net keeps STOCK real-FFT
    _spec/_magnitude/_mask/_ispec (torch.stft/istft) for the host side -- no
    conv-DFT kernels, so no 135 MB kernel matrix and no fp16-overflow DSP in or
    out of the graph, and the STFT is exact (not the 3.9e-4 conv approximation).
    """
    from demucs.pretrained import get_model
    bag = get_model(model_name)
    bag.eval()
    net = bag.models[0]
    net.eval()
    assert getattr(net, "cac", False), "split supports cac=True models"
    htdemucs_onnx.fix_localstate(net)       # EyeLike -> Equal, nothing else
    net.forward = types.MethodType(net_only_forward, net)
    return net, bag


def host_separate(net, mix):
    """Full torch pipeline with host-side real-FFT STFT/iSTFT around the net.
    net._spec/_ispec here are the STOCK torch.stft/istft methods (exact)."""
    length = mix.shape[-1]
    z = net._spec(mix)
    mag = net._magnitude(z)
    xspec_mask, xt = net(mag, mix)
    zout = net._mask(z, xspec_mask)
    xspec = net._ispec(zout, length)
    return xt + xspec
