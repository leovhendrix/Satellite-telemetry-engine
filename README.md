# satellite-telemetry-engine

A memory-safe, 100% Rust engine for satellite orbital telemetry processing and remote-sensing deep-learning ingestion. Built for embedded flight software and ground-station pipelines alike — no unsafe blocks, no unbounded allocations, explicit `Result`-based error handling throughout.

## What it does

**Orbital Telemetry Processor** (`src/telemetry.rs`)
- Converts between Cartesian state vectors (position/velocity) and classical Keplerian orbital elements.
- Solves Kepler's equation via Newton-Raphson with explicit convergence bounds.
- Propagates orbits under a combined **two-body gravity + J2 oblateness + exponential-atmosphere drag** acceleration model, integrated with classical 4th-order Runge-Kutta.
- Also exposes an analytic J2 *secular* propagator (nodal regression / apsidal precession / mean-motion correction) for fast coarse updates between full numerical passes.
- Detects orbital decay: propagation returns `AerospaceError::OrbitalDecayBelowSurface` the instant altitude crosses zero.

**Remote-Sensing Tensor Bridge** (`src/tensor_bridge.rs`)
- Validates and maps multi-band downlinked imaging packets into `ndarray` tensors (`(bands, height, width)`, batchable to `(batch, bands, height, width)`) ready for local inference.
- Per-band radiometric scaling and channel-wise standardization (zero-mean, unit-variance) built in.

**Async Ingestion Pipeline** (`src/ingestion.rs`)
- Non-blocking `tokio` producer/consumer pipeline over a bounded MPSC channel, so a slow consumer applies backpressure instead of stalling the runtime.

`src/main.rs` wires all three together into a runnable end-to-end demo.

## Design notes

- This is a **Simplified Perturbation Model** (two-body + J2 + drag), not a full SGP4/SDP4 implementation — it does not reproduce NORAD's TLE-fitted coefficient set. It's suited for first-order mission analysis, decay estimation, and as a numerically-integrated ground truth for testing higher-fidelity propagators.
- The exponential atmosphere model is a single-band fit; for production decay predictions, swap `AtmosphericModel` for a NRLMSISE-00 or JB2008 density model.
- Every fallible function returns `Result<T, AerospaceError>` — see `src/errors.rs` for the full taxonomy. Nothing panics on bad input.

## Run it

```bash
cargo test    # 14 unit tests: Kepler solver, round-trip conversions, drag physics, tensor mapping, async pipeline
cargo run --release
```

## Author

Built by Surafel Gashaw, a 16-year-old aerospace and software engineer passionate about orbital mechanics and high-performance systems programming.

## License

Dual-licensed under MIT or Apache-2.0.
