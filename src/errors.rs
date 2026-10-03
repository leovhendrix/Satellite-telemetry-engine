//! Centralized, explicit error taxonomy for the satellite telemetry engine.
//!
//! Every fallible operation in the orbital mechanics, remote-sensing tensor
//! bridge, and ingestion subsystems returns one of these variants rather than
//! panicking. This keeps the failure surface auditable end-to-end, which
//! matters when the same binary is expected to run unattended on embedded
//! satellite flight hardware.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum AerospaceError {
    #[error(
        "state vector contains a non-finite component (NaN or Inf) in position or velocity"
    )]
    InvalidStateVector,

    #[error(
        "orbital elements are physically invalid: {reason}"
    )]
    InvalidOrbitalElements { reason: String },

    #[error(
        "semi-major axis must be strictly positive for a bound elliptical orbit, got {semi_major_axis_meters} meters"
    )]
    NonPositiveSemiMajorAxis { semi_major_axis_meters: f64 },

    #[error(
        "eccentricity {eccentricity} is out of the supported range [0.0, 1.0) for closed-orbit propagation"
    )]
    UnsupportedEccentricity { eccentricity: f64 },

    #[error(
        "Kepler's equation failed to converge after {iterations_attempted} Newton-Raphson iterations (residual = {final_residual:e})"
    )]
    KeplerSolverDivergence {
        iterations_attempted: usize,
        final_residual: f64,
    },

    #[error("attempted matrix operation on a singular or ill-conditioned matrix: {context}")]
    SingularMatrix { context: String },

    #[error(
        "propagated altitude {altitude_meters} m fell below Earth's mean surface radius: satellite has decayed / re-entered"
    )]
    OrbitalDecayBelowSurface { altitude_meters: f64 },

    #[error("requested propagation duration {duration_seconds} s must be positive")]
    InvalidPropagationDuration { duration_seconds: f64 },

    #[error("requested integration time step {time_step_seconds} s must be positive and finite")]
    InvalidIntegrationStep { time_step_seconds: f64 },

    #[error("atmospheric density model rejected altitude {altitude_meters} m: {reason}")]
    AtmosphericModelOutOfBounds {
        altitude_meters: f64,
        reason: String,
    },

    #[error("spacecraft physical properties invalid: {reason}")]
    InvalidSpacecraftProperties { reason: String },

    #[error(
        "tensor shape mismatch while mapping telemetry into the deep-learning bridge: expected {expected:?}, got {actual:?}"
    )]
    TensorShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("telemetry packet failed validation: {reason}")]
    InvalidTelemetryPacket { reason: String },

    #[error("telemetry ingestion channel closed unexpectedly")]
    IngestionChannelClosed,
}
