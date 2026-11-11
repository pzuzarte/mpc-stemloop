#!/usr/bin/env python3
"""Make a synthetic 4-bar test loop so we can run the pipeline with no
external audio and no downloads. It mixes a kick, a bassline and a simple
melody into one stereo wav. Not musical art -- just a signal with clearly
separable layers to sanity-check the separation output.

ASCII-only stdout.
"""
import argparse
import numpy as np
import soundfile as sf


def kick(t, bpm):
    beat = 60.0 / bpm
    env_t = np.mod(t, beat)
    env = np.exp(-env_t * 35.0)
    pitch = 120.0 * np.exp(-env_t * 25.0) + 45.0
    return np.sin(2 * np.pi * pitch * env_t) * env


def bass(t, bpm):
    beat = 60.0 / bpm
    notes = [55.0, 55.0, 73.42, 65.41]  # A1 A1 D2 C2
    idx = (np.floor(t / beat).astype(int)) % len(notes)
    freq = np.array(notes)[idx]
    env = 0.6 + 0.4 * np.exp(-np.mod(t, beat) * 6.0)
    return 0.5 * np.sin(2 * np.pi * freq * t) * env


def lead(t, bpm):
    beat = 60.0 / bpm
    notes = [440.0, 523.25, 659.25, 587.33]  # A4 C5 E5 D5
    idx = (np.floor(t / (beat / 2)).astype(int)) % len(notes)
    freq = np.array(notes)[idx]
    env = np.exp(-np.mod(t, beat / 2) * 4.0)
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    return 0.25 * tone * env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../test-loops/test_loop.wav")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--bars", type=int, default=4)
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args()

    seconds = args.bars * 4 * 60.0 / args.bpm
    t = np.linspace(0, seconds, int(seconds * args.sr), endpoint=False)
    mix = kick(t, args.bpm) + bass(t, args.bpm) + lead(t, args.bpm)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.9
    stereo = np.stack([mix, mix], axis=1).astype(np.float32)
    sf.write(args.out, stereo, args.sr)
    print("wrote %s: %.2fs, %d Hz" % (args.out, seconds, args.sr))


if __name__ == "__main__":
    main()
