#!/usr/bin/env python3
"""Phase 1 prototype: separate a short audio loop into stems.

Goal of this phase is NOT to be fast or small. It is to prove the
loop -> stems pipeline works end to end and to measure the two numbers
that decide whether this is viable on the MPC Sample:

  * wall-clock time to separate a short loop
  * peak resident memory (RSS) during separation

Those numbers tell us whether a 2 GB / modest-ARM device has any chance.

Model: Demucs v4 (htdemucs) by default. It is heavy but high quality, so
it is the honest upper bound. Phase 2 swaps in a lighter / quantized model
and we compare against these baseline numbers.

Usage:
    python3 separate.py input.wav --out ../out --model htdemucs

ASCII-only stdout (latin-1 cluster locale rule).
"""
import argparse
import os
import resource
import sys
import time


def peak_rss_mb():
    # ru_maxrss is bytes on macOS, kilobytes on Linux. Normalise to MB.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / (1024 * 1024)
    return r / 1024


def main():
    ap = argparse.ArgumentParser(description="Separate a loop into stems.")
    ap.add_argument("input", help="input audio file (wav/flac/mp3)")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--model", default="htdemucs",
                    help="demucs model name (htdemucs, htdemucs_ft, mdx_extra, ...)")
    ap.add_argument("--shifts", type=int, default=0,
                    help="test-time shift augmentation passes (0 = fastest)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel jobs")
    ap.add_argument("--segment", type=float, default=None,
                    help="chunk length in seconds (smaller = lower peak memory)")
    ap.add_argument("--quantize", action="store_true",
                    help="dynamic INT8 quantization of Linear/LSTM layers")
    ap.add_argument("--no-write", action="store_true",
                    help="skip writing stems (benchmark the engine only)")
    args = ap.parse_args()

    # Imported here so the file can be read/scaffolded without the heavy deps.
    import torch
    import soundfile as sf
    import numpy as np
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    os.makedirs(args.out, exist_ok=True)

    t_load = time.time()
    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    # soundfile gives (frames, channels); demucs wants (channels, frames).
    wav = torch.from_numpy(audio.T.copy())
    dur = wav.shape[1] / sr
    print("loaded %s: %.2fs, %d ch, %d Hz (%.2fs)"
          % (os.path.basename(args.input), dur, wav.shape[0], sr,
             time.time() - t_load))

    torch.set_num_threads(max(1, args.jobs))

    t_model = time.time()
    model = get_model(args.model)
    model.eval()
    if args.quantize:
        # Dynamic INT8 covers Linear + LSTM (the transformer-heavy parts of
        # htdemucs). Conv layers are untouched, so this is a partial win.
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8)
    model_sr = model.samplerate
    print("model %s loaded: %d sources %s, sr=%d (%.2fs)"
          % (args.model, len(model.sources), model.sources, model_sr,
             time.time() - t_model))

    # Match the model's sample rate and channel count (demucs expects stereo).
    if sr != model_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, model_sr)
        sr = model_sr
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    ref = wav.mean(0)
    wav_in = (wav - ref.mean()) / (ref.std() + 1e-8)

    t_sep = time.time()
    with torch.no_grad():
        sources = apply_model(model, wav_in[None], shifts=args.shifts,
                              split=True, overlap=0.25, segment=args.segment,
                              progress=False, num_workers=args.jobs)[0]
    sep_time = time.time() - t_sep
    sources = sources * ref.std() + ref.mean()

    base = os.path.splitext(os.path.basename(args.input))[0]
    if not args.no_write:
        for name, source in zip(model.sources, sources):
            out_path = os.path.join(args.out, "%s_%s.wav" % (base, name))
            sf.write(out_path, source.T.numpy(), sr)
            print("wrote %s" % out_path)

    print("----")
    print("loop length   : %.2f s" % dur)
    print("separation    : %.2f s  (%.1fx realtime)"
          % (sep_time, sep_time / max(dur, 1e-6)))
    print("peak memory   : %.0f MB" % peak_rss_mb())
    print("----")
    print("Note: on a 2 GB ARM device, the stock MPC app already holds")
    print("~1 GB+. Compare peak memory above against the free headroom.")
    # Machine-readable line for bench.py.
    print("RESULT sep_s=%.2f peak_mb=%.0f" % (sep_time, peak_rss_mb()))


if __name__ == "__main__":
    main()
