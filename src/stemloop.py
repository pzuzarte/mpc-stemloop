#!/usr/bin/env python3
"""StemLoop: separate a short loop into drums/bass/other/vocals stems.

The real engine, assembled from the host-STFT split: the learned net runs in ONNX
Runtime; the STFT/iSTFT run on the host as exact real FFTs. Two host backends:

  --lean (default when the net ONNX is already cached): numpy FFT (host_dsp) + ORT
      only. No torch / demucs imported -> small RSS, the device-shaped path.
  full : torch real-FFT DSP + ORT. Also the path that EXPORTS a net ONNX for a new
      length bucket (export needs torch+demucs); after that, --lean can run it.

Pipeline (per short loop):
    mix (1,2,L) --[host STFT]--> mag
    (mag, mix)  --[ONNX net]---> xspec_mask, xt
    xspec_mask  --[host _mask + iSTFT]--> xspec
    stems = xt + xspec, trimmed to L --> one wav per source

Usage:
    python3 stemloop.py loop.wav --out ../out/loop            # auto lean if cached
    python3 stemloop.py loop.wav --out ../out/loop --full     # force torch path

ASCII-only stdout (latin-1 cluster locale rule).
"""
import argparse
import math
import os
import resource
import sys
import time

import numpy as np

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")

# demucs 4-stem source order (fixed for htdemucs / hdemucs_mmi).
SOURCES = {
    "hdemucs_mmi": ["drums", "bass", "other", "vocals"],
    "htdemucs": ["drums", "bass", "other", "vocals"],
}


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def net_onnx_path(model, secs):
    return os.path.join(MODELS, "%s_%ds_net.onnx" % (model, secs))


def load_audio_np(path):
    """soundfile-only loader -> (1,2,L) float32, sr. No torch, no resample."""
    import soundfile as sf
    x, sr = sf.read(path, always_2d=True, dtype="float32")   # (frames, chans)
    x = x.T
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    elif x.shape[0] > 2:
        x = x[:2]
    return np.ascontiguousarray(x)[None].astype("float32"), sr


def export_net_onnx(model, secs, length):
    """Export+cache the net-only ONNX for this bucket. Imports torch+demucs
    (only reached when the bucket is missing)."""
    import torch
    import split_export
    torch.backends.mha.set_fastpath_enabled(False)
    os.makedirs(MODELS, exist_ok=True)
    net, _ = split_export.prepare(model)
    mix = torch.zeros(1, 2, length)
    with torch.no_grad():
        z = net._spec(mix)
        mag = net._magnitude(z)
        torch.onnx.export(
            net, (mag, mix), net_onnx_path(model, secs),
            input_names=["mag", "mix"], output_names=["xspec", "xt"],
            opset_version=17, dynamo=False)
    return net_onnx_path(model, secs)


def run_ort(onnx_path, mag, mix, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    t = time.time()
    xspec, xt = sess.run(None, {names[0]: mag.astype("float32"),
                                names[1]: mix.astype("float32")})
    return xspec, xt, time.time() - t


def separate(in_wav, out_dir, model="hdemucs_mmi", max_seconds=12, threads=4,
             force_full=False):
    import soundfile as sf
    mix, in_sr = load_audio_np(in_wav)
    if in_sr != SR:
        # Resampling needs torch(audio); keep the lean path pure. Ask for 44.1k.
        raise SystemExit("input is %d Hz; please provide 44100 Hz audio "
                         "(resample upstream)." % in_sr)
    L = mix.shape[-1]
    secs = min(max(int(math.ceil(L / SR)), 1), max_seconds)
    tgt = secs * SR
    if L <= tgt:
        mix_p = np.pad(mix, [(0, 0), (0, 0), (0, tgt - L)])
        out_len = L
    else:
        mix_p = mix[..., :tgt]
        out_len = tgt
        print("note: loop > %ds, truncated" % max_seconds)

    onnx_path = net_onnx_path(model, secs)
    exported = False
    if not os.path.exists(onnx_path):
        print("net ONNX for %ds not cached -- exporting (one-time, needs torch)" % secs)
        export_net_onnx(model, secs, tgt)
        exported = True

    lean = not force_full
    t0 = time.time()
    if lean:
        import host_dsp
        zr, zi = host_dsp.spec_np(mix_p)
        mag = host_dsp.magnitude_np(zr, zi)
        xspec, xt, net_s = run_ort(onnx_path, mag, mix_p, threads)
        stems = host_dsp.mask_ispec_np(xspec, out_len)[0] + xt[0][:, :, :out_len]
        backend = "lean (numpy + ORT)"
    else:
        import torch
        import split_export
        net, _ = split_export.prepare(model)
        with torch.no_grad():
            z = net._spec(torch.from_numpy(mix_p))
            mag = net._magnitude(z).numpy()
            xspec, xt, net_s = run_ort(onnx_path, mag, mix_p, threads)
            zout = net._mask(z, torch.from_numpy(xspec))
            xs = net._ispec(zout, tgt).numpy()
        stems = (xt + xs)[0, :, :, :out_len]
        backend = "full (torch + ORT)"
    total_s = time.time() - t0

    sources = SOURCES.get(model, ["drums", "bass", "other", "vocals"])
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i, name in enumerate(sources):
        wav = stems[i].T                                   # (L, 2)
        p = os.path.join(out_dir, "%s.wav" % name)
        sf.write(p, wav, SR)
        written.append((name, p, float(np.abs(wav).max())))

    print("input    : %s  (sr=%d, %d samples = %.2fs)"
          % (os.path.basename(in_wav), SR, L, L / SR))
    print("model    : %s  net ONNX at %ds bucket%s"
          % (model, secs, " (just exported)" if exported else ""))
    print("backend  : %s" % backend)
    print("timing   : total %.2fs  (ort net %.2fs)" % (total_s, net_s))
    print("peak RSS : %.0f MB" % peak_mb())
    for name, p, pk in written:
        print("  wrote %-8s peak %.3f  %s" % (name, pk, p))
    return written


def main():
    ap = argparse.ArgumentParser(description="Separate a short loop into stems.")
    ap.add_argument("input", help="input loop (wav/flac), 44.1 kHz")
    ap.add_argument("--out", default="../out/stemloop", help="output directory")
    ap.add_argument("--model", default="hdemucs_mmi", help="demucs model name")
    ap.add_argument("--max-seconds", type=int, default=12,
                    help="cap for 'short' loops (longer inputs are truncated)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--full", action="store_true",
                    help="use the torch DSP backend instead of the lean numpy one")
    args = ap.parse_args()
    separate(args.input, args.out, args.model, args.max_seconds, args.threads,
             force_full=args.full)


if __name__ == "__main__":
    main()
