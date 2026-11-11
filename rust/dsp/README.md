# mpc-stemloop-dsp

Host STFT/iSTFT for the MPC Sample stem-split mod, a Rust port of
`../../src/host_dsp.py`. Kept bit-parity with demucs `HDemucs._spec` / `_ispec`
so the split pipeline can run the learned net in an inference runtime with the
transform done here (no torch/demucs at run time).

This crate is host/`std` Rust; it is the DSP half of the on-device engine
(`rust/engine`), kept separate so its numpy parity can be tested on its own.

## API

- `Dsp::new()` -- cached FFT plans + window for the model's `NFFT`.
- `Dsp::stft_row` / `istft_row` -- single-row STFT/iSTFT, `(F, T)` layout.
- `spec(mix) -> (zr, zi, le)` -- `_spec`.
- `magnitude(zr, zi) -> mag` -- `_magnitude` (complex-as-channels).
- `mask_ispec(xspec, length) -> stems` -- `_mask` + `_ispec`.

Params are fixed by the model: `NFFT = 4096`, `HOP = 1024`, periodic Hann,
center, normalized. Internals compute in `f64` to match numpy; I/O is `f32`.

## Build and test

```sh
# regenerate the parity fixtures from the numpy reference (once / when host_dsp
# changes); needs the project venv with numpy.
../../.venv/bin/python ../../src/gen_dsp_fixtures.py

cargo test --release            # 4 parity tests vs the numpy fixtures
cargo clippy --release --lib -- -W clippy::all -W clippy::pedantic
cargo fmt --check
```

Current parity vs numpy: every array matches to `0.0` except the STFT imaginary
part at `2.6e-16` (f64 round-off) -- effectively bit-exact, well under the
`1e-6` target.

The `tests/fixtures/*.bin` files are generated and git-ignored; run the
generator above before `cargo test` on a fresh checkout.
