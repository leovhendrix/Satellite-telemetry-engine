//! Satellite Telemetry & Remote Sensing Processing Engine
//!
//! Coordinates three subsystems:
//!   1. `telemetry`     — orbital mechanics: state vectors, Keplerian
//!                        elements, J2 secular propagation, and numerically
//!                        integrated gravity + drag decay propagation.
//!   2. `tensor_bridge` — maps downlinked multi-band remote-sensing packets
//!                        into `ndarray` tensors ready for local inference.
//!   3. `ingestion`      — a non-blocking `tokio` pipeline simulating a
//!                        real-time telemetry downlink.

mod errors;
mod ingestion;
mod telemetry;
mod tensor_bridge;

use errors::AerospaceError;
use telemetry::{
    propagate_orbit_with_drag_decay, state_vector_to_keplerian_elements, AtmosphericModel,
    KeplerianElements, SpacecraftPhysicalProperties, StateVector,
};
use tensor_bridge::map_packet_to_chw_tensor;
use tokio::time::Duration;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    info!("satellite telemetry & remote sensing processing engine starting up");

    run_orbital_propagation_demo()?;
    run_telemetry_ingestion_pipeline().await?;

    info!("engine run complete");
    Ok(())
}

/// Demonstrates the Orbital Telemetry Processor: builds a representative
/// low-Earth-orbit state vector, converts it to Keplerian elements, and
/// numerically propagates it forward under combined gravity, J2 oblateness,
/// and atmospheric drag until either the requested duration elapses or the
/// satellite decays below the Earth's surface.
fn run_orbital_propagation_demo() -> Result<(), AerospaceError> {
    let epoch = chrono::Utc::now();

    // A representative 500 km sun-synchronous-like circular orbit.
    let initial_elements = KeplerianElements {
        semi_major_axis_meters: telemetry::EARTH_EQUATORIAL_RADIUS_METERS + 500_000.0,
        eccentricity: 0.001,
        inclination_radians: 97.4_f64.to_radians(),
        right_ascension_of_ascending_node_radians: 15.0_f64.to_radians(),
        argument_of_perigee_radians: 60.0_f64.to_radians(),
        mean_anomaly_radians: 0.0,
        epoch,
    };

    let initial_state = telemetry::keplerian_elements_to_state_vector(&initial_elements)?;

    info!(
        position_km = ?initial_state.position_meters.map(|c| c / 1000.0),
        velocity_km_s = ?initial_state.velocity_meters_per_second.map(|c| c / 1000.0),
        altitude_km = initial_state.altitude_meters() / 1000.0,
        "computed initial state vector from Keplerian elements"
    );

    let recovered_elements = state_vector_to_keplerian_elements(&initial_state)?;
    info!(
        semi_major_axis_km = recovered_elements.semi_major_axis_meters / 1000.0,
        eccentricity = recovered_elements.eccentricity,
        inclination_deg = recovered_elements.inclination_radians.to_degrees(),
        "round-tripped state vector back to Keplerian elements"
    );

    // A small imaging cubesat: 12 kg, Cd = 2.2, 0.03 m^2 ram-facing area —
    // deliberately high area-to-mass ratio so drag decay is visible over a
    // short simulated horizon in this demonstration.
    let spacecraft = SpacecraftPhysicalProperties::new(12.0, 2.2, 0.03)?;
    let atmosphere = AtmosphericModel::low_earth_orbit_default();

    let propagation_duration_seconds = 3.0 * 60.0 * 60.0; // 3 hours
    let integration_time_step_seconds = 30.0;

    match propagate_orbit_with_drag_decay(
        &initial_state,
        &spacecraft,
        &atmosphere,
        propagation_duration_seconds,
        integration_time_step_seconds,
    ) {
        Ok(trajectory) => {
            let final_state: &StateVector = trajectory
                .last()
                .expect("trajectory always contains at least the initial state");
            info!(
                samples = trajectory.len(),
                final_altitude_km = final_state.altitude_meters() / 1000.0,
                "orbit propagation with J2 + drag completed without decay"
            );
        }
        Err(AerospaceError::OrbitalDecayBelowSurface { altitude_meters }) => {
            warn!(
                altitude_meters,
                "satellite decayed below the Earth's surface during propagation window"
            );
        }
        Err(propagation_error) => return Err(propagation_error),
    }

    // Demonstrate a raw two-body + drag acceleration query at the initial
    // state, independent of the full trajectory integration above.
    let drag_acceleration = telemetry::compute_atmospheric_drag_acceleration(
        &initial_state,
        &spacecraft,
        &atmosphere,
    )?;
    info!(
        drag_acceleration_m_s2 = ?drag_acceleration,
        "instantaneous atmospheric drag acceleration at initial epoch"
    );

    Ok(())
}

/// Demonstrates the ingestion + tensor-bridge subsystems: spins up the async
/// telemetry pipeline, consumes packets as they arrive without blocking,
/// and maps each one into an inference-ready tensor.
async fn run_telemetry_ingestion_pipeline() -> anyhow::Result<()> {
    let (mut packet_receiver, producer_handle) =
        ingestion::spawn_ingestion_pipeline(Duration::from_millis(50), 10);

    let mut packets_processed: u64 = 0;

    while let Some(packet) = packet_receiver.recv().await {
        match map_packet_to_chw_tensor(&packet) {
            Ok(mut tensor) => {
                if let Err(standardization_error) =
                    tensor_bridge::standardize_channels_in_place(&mut tensor)
                {
                    warn!(
                        packet_sequence_number = packet.packet_sequence_number,
                        error = %standardization_error,
                        "skipping standardization for this frame"
                    );
                }

                info!(
                    packet_sequence_number = packet.packet_sequence_number,
                    tensor_shape = ?tensor.shape(),
                    "mapped remote-sensing packet into inference-ready tensor"
                );
                packets_processed += 1;
            }
            Err(mapping_error) => {
                error!(
                    packet_sequence_number = packet.packet_sequence_number,
                    error = %mapping_error,
                    "failed to map telemetry packet into tensor"
                );
            }
        }
    }

    producer_handle.await??;
    info!(packets_processed, "telemetry ingestion pipeline drained");

    Ok(())
}
