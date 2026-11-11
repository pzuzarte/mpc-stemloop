#!/usr/bin/env python3
"""Phase 2 benchmark: run separate.py under several memory configs, each in
its OWN fresh process, so peak RSS (a per-process high-water mark) is
comparable across configs. Prints one table.

We vary:
  * model     -- htdemucs (transformer, heavy) vs mdx_extra_q / hdemucs_mmi
                 (lighter / quantized conv models)
  * segment   -- smaller chunks lower the peak at some speed cost
  * quantize  -- dynamic INT8 on Linear/LSTM layers

Env per run mirrors the device as far as macOS allows:
  MALLOC_ARENA_MAX=1 (no-op on macOS, real win on glibc/ARM) and a fixed
  thread count to reduce variance.

Caveat: absolute MB here is NOT the on-device number -- torch's macOS
runtime overhead differs from a cross-compiled ARM build. The RANKING
between configs is what transfers. ASCII-only output.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "..", "test-loops", "test_loop.wav")

CONFIGS = [
    ("htdemucs full (baseline)", ["--model", "htdemucs"]),
    ("htdemucs seg=3",           ["--model", "htdemucs", "--segment", "3"]),
    ("htdemucs seg=3 +quant",    ["--model", "htdemucs", "--segment", "3",
                                  "--quantize"]),
    ("hdemucs_mmi seg=3",        ["--model", "hdemucs_mmi", "--segment", "3"]),
    ("mdx_extra_q full",         ["--model", "mdx_extra_q"]),
    ("mdx_extra_q seg=3",        ["--model", "mdx_extra_q", "--segment", "3"]),
]

RESULT_RE = re.compile(r"RESULT sep_s=([0-9.]+) peak_mb=([0-9.]+)")


def run(args):
    env = dict(os.environ)
    env["MALLOC_ARENA_MAX"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    cmd = [sys.executable, os.path.join(HERE, "separate.py"), INPUT,
           "--no-write", "--jobs", "4"] + args
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    m = RESULT_RE.search(p.stdout)
    if not m:
        tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
        return None, None, " | ".join(tail)
    return float(m.group(1)), float(m.group(2)), ""


def main():
    rows = []
    for name, args in CONFIGS:
        print("running: %s ..." % name, flush=True)
        sep, mem, err = run(args)
        if sep is None:
            print("  FAILED: %s" % err, flush=True)
            rows.append((name, None, None))
        else:
            print("  %.2f s, %.0f MB" % (sep, mem), flush=True)
            rows.append((name, sep, mem))

    print()
    print("%-28s %10s %12s" % ("config", "time (s)", "peak (MB)"))
    print("-" * 52)
    for name, sep, mem in rows:
        if sep is None:
            print("%-28s %10s %12s" % (name, "FAIL", "-"))
        else:
            print("%-28s %10.2f %12.0f" % (name, sep, mem))
    print("-" * 52)
    print("Target for 2 GB device: peak well under ~500 MB.")


if __name__ == "__main__":
    main()
