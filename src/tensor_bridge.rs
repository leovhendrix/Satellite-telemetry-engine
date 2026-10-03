//! Remote-Sensing Deep Learning Tensor Bridge
//!
//! Maps raw, multi-band geospatial telemetry packets (as they arrive off the
//! downlink) into structured, contiguous `ndarray` tensors suitable for
//! feeding directly into a local inference engine (e.g. an ONNX Runtime or
//! `tch` session running on the satellite's onboard compute module, or a
//! ground-station GPU pipeline). This module owns the packet validation and
//! shape-mapping logic; it does not perform inference itself, keeping the
//! bridge decoupled from any specific model runtime.

use crate::errors::AerospaceError;
use ndarray::{Array3, Array4};
use serde::{Deserialize, Serialize};

/// A single downlinked remote-sensing telemetry packet: one imaging frame
/// composed of several spectral bands, each a flattened row-major raster.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteSensingPacket {
    pub packet_sequence_number: u64,
    pub spectral_band_count: usize,
    pub raster_height_pixels: usize,
    pub raster_width_pixels: usize,
    /// Flattened as [band][row][col] in row-major order, one Vec per band.
    pub spectral_bands: Vec<Vec<f32>>,
    /// Per-band radiometric scale factors applied by the sensor's onboard
    /// calibration stage prior to downlink.
    pub radiometric_scale_factors: Vec<f32>,
}

impl RemoteSensingPacket {
    pub fn validate(&self) -> Result<(), AerospaceError> {
        if self.spectral_bands.len() != self.spectral_band_count {
            return Err(AerospaceError::InvalidTelemetryPacket {
                reason: format!(
                    "declared spectral_band_count {} does not match {} band arrays actually present",
                    self.spectral_band_count,
                    self.spectral_bands.len()
                ),
            });
        }

        if self.radiometric_scale_factors.len() != self.spectral_band_count {
            return Err(AerospaceError::InvalidTelemetryPacket {
                reason: format!(
                    "expected {} radiometric scale factors, found {}",
                    self.spectral_band_count,
                    self.radiometric_scale_factors.len()
                ),
            });
        }

        let expected_pixels_per_band = self.raster_height_pixels * self.raster_width_pixels;
        for (band_index, band_pixels) in self.spectral_bands.iter().enumerate() {
            if band_pixels.len() != expected_pixels_per_band {
                return Err(AerospaceError::InvalidTelemetryPacket {
                    reason: format!(
                        "band {band_index} has {} pixels, expected {expected_pixels_per_band} ({}x{})",
                        band_pixels.len(),
                        self.raster_height_pixels,
                        self.raster_width_pixels
                    ),
                });
            }
            if band_pixels.iter().any(|pixel| !pixel.is_finite()) {
                return Err(AerospaceError::InvalidTelemetryPacket {
                    reason: format!("band {band_index} contains a non-finite pixel value"),
                });
            }
        }

        Ok(())
    }
}

/// Maps a single validated `RemoteSensingPacket` into a channels-first
/// `Array3<f32>` tensor of shape `(bands, height, width)`, applying each
/// band's radiometric scale factor during the copy so the resulting tensor
/// is inference-ready without a separate normalization pass.
pub fn map_packet_to_chw_tensor(packet: &RemoteSensingPacket) -> Result<Array3<f32>, AerospaceError> {
    packet.validate()?;

    let mut tensor = Array3::<f32>::zeros((
        packet.spectral_band_count,
        packet.raster_height_pixels,
        packet.raster_width_pixels,
    ));

    for band_index in 0..packet.spectral_band_count {
        let scale_factor = packet.radiometric_scale_factors[band_index];
        let band_pixels = &packet.spectral_bands[band_index];

        for row_index in 0..packet.raster_height_pixels {
            let row_start_offset = row_index * packet.raster_width_pixels;
            for col_index in 0..packet.raster_width_pixels {
                let raw_pixel_value = band_pixels[row_start_offset + col_index];
                tensor[[band_index, row_index, col_index]] = raw_pixel_value * scale_factor;
            }
        }
    }

    Ok(tensor)
}

/// Batches a slice of already-mapped `Array3<f32>` (bands, height, width)
/// tensors into a single `Array4<f32>` of shape
/// `(batch, bands, height, width)` for batched model inference. All frames
/// in the batch must share identical spatial and spectral dimensions.
pub fn batch_chw_tensors(frames: &[Array3<f32>]) -> Result<Array4<f32>, AerospaceError> {
    let first_frame = frames.first().ok_or_else(|| AerospaceError::InvalidTelemetryPacket {
        reason: "cannot batch an empty set of tensor frames".to_string(),
    })?;

    let expected_shape = first_frame.shape().to_vec();

    for (frame_index, frame) in frames.iter().enumerate() {
        if frame.shape() != expected_shape.as_slice() {
            return Err(AerospaceError::TensorShapeMismatch {
                expected: expected_shape.clone(),
                actual: frame.shape().to_vec(),
            });
        }
        let _ = frame_index;
    }

    let batch_size = frames.len();
    let (band_count, raster_height, raster_width) =
        (expected_shape[0], expected_shape[1], expected_shape[2]);

    let mut batched_tensor = Array4::<f32>::zeros((batch_size, band_count, raster_height, raster_width));

    for (frame_index, frame) in frames.iter().enumerate() {
        batched_tensor
            .index_axis_mut(ndarray::Axis(0), frame_index)
            .assign(frame);
    }

    Ok(batched_tensor)
}

/// Applies channel-wise standardization (zero mean, unit variance) in place,
/// the conventional pre-inference normalization step for CNN-based
/// localized inference models running on downlinked imagery.
pub fn standardize_channels_in_place(tensor: &mut Array3<f32>) -> Result<(), AerospaceError> {
    let band_count = tensor.shape()[0];

    for band_index in 0..band_count {
        let mut band_view = tensor.index_axis_mut(ndarray::Axis(0), band_index);

        let pixel_count = band_view.len() as f32;
        if pixel_count == 0.0 {
            return Err(AerospaceError::InvalidTelemetryPacket {
                reason: format!("band {band_index} has zero pixels; cannot standardize"),
            });
        }

        let band_mean = band_view.sum() / pixel_count;
        let band_variance = band_view.iter().map(|pixel| (pixel - band_mean).powi(2)).sum::<f32>()
            / pixel_count;
        let band_standard_deviation = band_variance.sqrt();

        if band_standard_deviation < f32::EPSILON {
            return Err(AerospaceError::InvalidTelemetryPacket {
                reason: format!(
                    "band {band_index} has near-zero variance; standardization would divide by zero"
                ),
            });
        }

        band_view.mapv_inplace(|pixel| (pixel - band_mean) / band_standard_deviation);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_two_band_packet() -> RemoteSensingPacket {
        RemoteSensingPacket {
            packet_sequence_number: 1,
            spectral_band_count: 2,
            raster_height_pixels: 2,
            raster_width_pixels: 2,
            spectral_bands: vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![10.0, 20.0, 30.0, 40.0],
            ],
            radiometric_scale_factors: vec![1.0, 0.5],
        }
    }

    #[test]
    fn packet_validation_rejects_shape_mismatch() {
        let mut packet = synthetic_two_band_packet();
        packet.spectral_bands[0].pop();
        assert!(packet.validate().is_err());
    }

    #[test]
    fn mapping_applies_radiometric_scale_factors() {
        let packet = synthetic_two_band_packet();
        let tensor = map_packet_to_chw_tensor(&packet).unwrap();

        assert_eq!(tensor.shape(), &[2, 2, 2]);
        assert_eq!(tensor[[1, 0, 0]], 5.0); // 10.0 * 0.5
    }

    #[test]
    fn batching_requires_matching_shapes() {
        let packet = synthetic_two_band_packet();
        let frame = map_packet_to_chw_tensor(&packet).unwrap();
        let mismatched_frame = Array3::<f32>::zeros((3, 2, 2));

        let result = batch_chw_tensors(&[frame, mismatched_frame]);
        assert!(matches!(result, Err(AerospaceError::TensorShapeMismatch { .. })));
    }

    #[test]
    fn standardization_yields_zero_mean_per_band() {
        let packet = synthetic_two_band_packet();
        let mut tensor = map_packet_to_chw_tensor(&packet).unwrap();
        standardize_channels_in_place(&mut tensor).unwrap();

        for band_index in 0..2 {
            let band_view = tensor.index_axis(ndarray::Axis(0), band_index);
            let band_mean = band_view.sum() / band_view.len() as f32;
            assert!(band_mean.abs() < 1e-5);
        }
    }
}
