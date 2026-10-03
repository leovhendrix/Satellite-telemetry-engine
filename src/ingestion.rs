//! Async Telemetry Ingestion Pipeline
//!
//! Provides a non-blocking producer/consumer pipeline for streaming
//! telemetry packets off a (simulated) downlink socket into the processing
//! engine. The producer and consumer run as independent `tokio` tasks
//! connected by a bounded MPSC channel, so a slow consumer (e.g. one doing
//! heavy tensor-mapping work) applies natural backpressure to the ingestion
//! side without blocking the async runtime's worker threads.

use crate::errors::AerospaceError;
use crate::tensor_bridge::RemoteSensingPacket;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::{info, warn};

/// Channel capacity before the ingestion producer starts applying
/// backpressure to the simulated downlink source.
const INGESTION_CHANNEL_CAPACITY: usize = 64;

/// Spawns the ingestion producer task, which emits synthetic
/// `RemoteSensingPacket`s at a fixed cadence to simulate a real-time
/// downlink stream. Returns the receiving half of the channel plus a
/// `JoinHandle` the caller can await for graceful shutdown.
pub fn spawn_ingestion_pipeline(
    packet_emission_period: Duration,
    total_packets_to_emit: u64,
) -> (
    mpsc::Receiver<RemoteSensingPacket>,
    tokio::task::JoinHandle<Result<(), AerospaceError>>,
) {
    let (packet_sender, packet_receiver) = mpsc::channel(INGESTION_CHANNEL_CAPACITY);

    let producer_handle = tokio::spawn(async move {
        let mut emission_interval = interval(packet_emission_period);

        for packet_sequence_number in 0..total_packets_to_emit {
            emission_interval.tick().await;

            let synthetic_packet = generate_synthetic_remote_sensing_packet(packet_sequence_number);

            if let Err(send_error) = packet_sender.send(synthetic_packet).await {
                warn!(
                    error = %send_error,
                    "ingestion consumer dropped; stopping producer early"
                );
                return Err(AerospaceError::IngestionChannelClosed);
            }
        }

        info!(total_packets_to_emit, "ingestion pipeline completed emission");
        Ok(())
    });

    (packet_receiver, producer_handle)
}

/// Builds a small synthetic 4-band packet (mimicking, e.g., visible RGB +
/// near-infrared bands from an optical remote-sensing payload) so the
/// pipeline is runnable end-to-end without a real downlink connection.
fn generate_synthetic_remote_sensing_packet(packet_sequence_number: u64) -> RemoteSensingPacket {
    const RASTER_HEIGHT_PIXELS: usize = 8;
    const RASTER_WIDTH_PIXELS: usize = 8;
    const SPECTRAL_BAND_COUNT: usize = 4;

    let spectral_bands: Vec<Vec<f32>> = (0..SPECTRAL_BAND_COUNT)
        .map(|band_index| {
            (0..(RASTER_HEIGHT_PIXELS * RASTER_WIDTH_PIXELS))
                .map(|pixel_index| {
                    let phase = (packet_sequence_number as f32) * 0.1
                        + (band_index as f32) * 0.3
                        + (pixel_index as f32) * 0.05;
                    100.0 + 50.0 * phase.sin()
                })
                .collect()
        })
        .collect();

    RemoteSensingPacket {
        packet_sequence_number,
        spectral_band_count: SPECTRAL_BAND_COUNT,
        raster_height_pixels: RASTER_HEIGHT_PIXELS,
        raster_width_pixels: RASTER_WIDTH_PIXELS,
        spectral_bands,
        radiometric_scale_factors: vec![1.0; SPECTRAL_BAND_COUNT],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pipeline_emits_the_requested_packet_count() {
        let (mut receiver, producer_handle) =
            spawn_ingestion_pipeline(Duration::from_millis(1), 5);

        let mut received_packets = Vec::new();
        while let Some(packet) = receiver.recv().await {
            received_packets.push(packet);
        }

        producer_handle.await.unwrap().unwrap();
        assert_eq!(received_packets.len(), 5);
    }

    #[tokio::test]
    async fn synthetic_packets_pass_validation() {
        let packet = generate_synthetic_remote_sensing_packet(0);
        assert!(packet.validate().is_ok());
    }
}
