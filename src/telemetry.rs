//! Orbital Telemetry Processor
//!
//! Implements a Simplified Perturbation Model (two-body dynamics plus J2
//! oblateness secular perturbations and exponential-atmosphere drag decay)
//! for computing and propagating satellite state vectors. This module is
//! deliberately independent from any specific packet format: it operates on
//! physical quantities (meters, seconds, radians, kilograms) so it can sit
//! behind any ingestion front-end.
//!
//! Reference frame convention: all Cartesian position/velocity vectors are
//! expressed in the Earth-Centered Inertial (ECI) frame unless a function
//! name explicitly says otherwise (e.g. `eci_to_ecef_rotation_matrix`).

use crate::errors::AerospaceError;
use chrono::{DateTime, Utc};
use nalgebra::{Matrix3, Vector3};

// ---------------------------------------------------------------------------
// Physical constants (WGS84 / IAU / US Standard Atmosphere 1976)
// ---------------------------------------------------------------------------

/// Earth's standard gravitational parameter, mu = G * M_earth, in m^3 / s^2.
pub const EARTH_GRAVITATIONAL_PARAMETER_M3_S2: f64 = 3.986_004_418e14;

/// WGS84 equatorial radius of the Earth, in meters.
pub const EARTH_EQUATORIAL_RADIUS_METERS: f64 = 6_378_137.0;

/// Earth's second dynamic form factor (oblateness coefficient), dimensionless.
pub const EARTH_J2_OBLATENESS_COEFFICIENT: f64 = 1.082_626_68e-3;

/// Earth's mean sidereal angular rotation rate, in rad/s.
pub const EARTH_ANGULAR_ROTATION_RATE_RAD_S: f64 = 7.292_115_0e-5;

/// Newton-Raphson convergence tolerance for Kepler's equation (radians).
const KEPLER_SOLVER_TOLERANCE_RADIANS: f64 = 1e-12;

/// Hard cap on Newton-Raphson iterations before declaring divergence.
const KEPLER_SOLVER_MAX_ITERATIONS: usize = 100;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A Cartesian orbital state at a specific epoch, expressed in the ECI frame.
#[derive(Debug, Clone, PartialEq)]
pub struct StateVector {
    pub position_meters: Vector3<f64>,
    pub velocity_meters_per_second: Vector3<f64>,
    pub epoch: DateTime<Utc>,
}

impl StateVector {
    pub fn new(
        position_meters: Vector3<f64>,
        velocity_meters_per_second: Vector3<f64>,
        epoch: DateTime<Utc>,
    ) -> Result<Self, AerospaceError> {
        if !position_meters.iter().all(|component| component.is_finite())
            || !velocity_meters_per_second
                .iter()
                .all(|component| component.is_finite())
        {
            return Err(AerospaceError::InvalidStateVector);
        }
        Ok(Self {
            position_meters,
            velocity_meters_per_second,
            epoch,
        })
    }

    /// Geocentric altitude above the WGS84 equatorial radius, in meters.
    /// This is a spherical-Earth approximation, adequate for drag and
    /// decay-threshold checks but not for high-precision geodesy.
    pub fn altitude_meters(&self) -> f64 {
        self.position_meters.norm() - EARTH_EQUATORIAL_RADIUS_METERS
    }
}

/// Classical Keplerian orbital elements.
#[derive(Debug, Clone, PartialEq)]
pub struct KeplerianElements {
    pub semi_major_axis_meters: f64,
    pub eccentricity: f64,
    pub inclination_radians: f64,
    pub right_ascension_of_ascending_node_radians: f64,
    pub argument_of_perigee_radians: f64,
    pub mean_anomaly_radians: f64,
    pub epoch: DateTime<Utc>,
}

impl KeplerianElements {
    fn validate(&self) -> Result<(), AerospaceError> {
        if !self.semi_major_axis_meters.is_finite() || self.semi_major_axis_meters <= 0.0 {
            return Err(AerospaceError::NonPositiveSemiMajorAxis {
                semi_major_axis_meters: self.semi_major_axis_meters,
            });
        }
        if !(0.0..1.0).contains(&self.eccentricity) {
            return Err(AerospaceError::UnsupportedEccentricity {
                eccentricity: self.eccentricity,
            });
        }
        if !self.inclination_radians.is_finite()
            || !self.right_ascension_of_ascending_node_radians.is_finite()
            || !self.argument_of_perigee_radians.is_finite()
            || !self.mean_anomaly_radians.is_finite()
        {
            return Err(AerospaceError::InvalidOrbitalElements {
                reason: "one or more angular elements is non-finite".to_string(),
            });
        }
        Ok(())
    }

    /// Mean motion, n = sqrt(mu / a^3), in rad/s.
    pub fn mean_motion_rad_s(&self) -> f64 {
        (EARTH_GRAVITATIONAL_PARAMETER_M3_S2 / self.semi_major_axis_meters.powi(3)).sqrt()
    }
}

/// Physical properties of the spacecraft bus relevant to drag modeling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpacecraftPhysicalProperties {
    pub dry_mass_kilograms: f64,
    pub drag_coefficient: f64,
    pub ram_facing_cross_sectional_area_m2: f64,
}

impl SpacecraftPhysicalProperties {
    pub fn new(
        dry_mass_kilograms: f64,
        drag_coefficient: f64,
        ram_facing_cross_sectional_area_m2: f64,
    ) -> Result<Self, AerospaceError> {
        if dry_mass_kilograms <= 0.0 {
            return Err(AerospaceError::InvalidSpacecraftProperties {
                reason: format!("dry mass must be positive, got {dry_mass_kilograms} kg"),
            });
        }
        if drag_coefficient <= 0.0 {
            return Err(AerospaceError::InvalidSpacecraftProperties {
                reason: format!(
                    "drag coefficient must be positive, got {drag_coefficient}"
                ),
            });
        }
        if ram_facing_cross_sectional_area_m2 <= 0.0 {
            return Err(AerospaceError::InvalidSpacecraftProperties {
                reason: format!(
                    "cross-sectional area must be positive, got {ram_facing_cross_sectional_area_m2} m^2"
                ),
            });
        }
        Ok(Self {
            dry_mass_kilograms,
            drag_coefficient,
            ram_facing_cross_sectional_area_m2,
        })
    }

    /// Ballistic coefficient term (Cd * A / m), in m^2/kg.
    pub fn ballistic_area_to_mass_ratio(&self) -> f64 {
        (self.drag_coefficient * self.ram_facing_cross_sectional_area_m2)
            / self.dry_mass_kilograms
    }
}

/// Exponential atmospheric density model, piecewise-fit around a reference
/// altitude band (consistent in structure with the US Standard Atmosphere
/// 1976 exponential approximation used for first-order drag estimates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphericModel {
    pub reference_altitude_meters: f64,
    pub reference_density_kg_m3: f64,
    pub scale_height_meters: f64,
}

impl AtmosphericModel {
    /// A reasonable default tuned for the 400-600 km low-Earth-orbit band,
    /// where most operational remote-sensing constellations fly.
    pub fn low_earth_orbit_default() -> Self {
        Self {
            reference_altitude_meters: 400_000.0,
            reference_density_kg_m3: 5.464e-13,
            scale_height_meters: 58_515.0,
        }
    }

    pub fn new(
        reference_altitude_meters: f64,
        reference_density_kg_m3: f64,
        scale_height_meters: f64,
    ) -> Result<Self, AerospaceError> {
        if scale_height_meters <= 0.0 {
            return Err(AerospaceError::InvalidSpacecraftProperties {
                reason: format!(
                    "atmospheric scale height must be positive, got {scale_height_meters} m"
                ),
            });
        }
        if reference_density_kg_m3 < 0.0 {
            return Err(AerospaceError::AtmosphericModelOutOfBounds {
                altitude_meters: reference_altitude_meters,
                reason: "reference density cannot be negative".to_string(),
            });
        }
        Ok(Self {
            reference_altitude_meters,
            reference_density_kg_m3,
            scale_height_meters,
        })
    }

    /// rho(h) = rho_ref * exp(-(h - h_ref) / H)
    pub fn density_at_altitude(&self, altitude_meters: f64) -> Result<f64, AerospaceError> {
        if altitude_meters <= -EARTH_EQUATORIAL_RADIUS_METERS {
            return Err(AerospaceError::AtmosphericModelOutOfBounds {
                altitude_meters,
                reason: "altitude is below the center of the Earth".to_string(),
            });
        }
        let altitude_delta_meters = altitude_meters - self.reference_altitude_meters;
        let density = self.reference_density_kg_m3
            * (-altitude_delta_meters / self.scale_height_meters).exp();
        if !density.is_finite() {
            return Err(AerospaceError::AtmosphericModelOutOfBounds {
                altitude_meters,
                reason: "exponential density model overflowed to a non-finite value".to_string(),
            });
        }
        Ok(density)
    }
}

// ---------------------------------------------------------------------------
// Rotation matrices (elementary + composite)
// ---------------------------------------------------------------------------

/// Elementary rotation about the X-axis by `angle_radians`.
pub fn rotation_matrix_about_x_axis(angle_radians: f64) -> Matrix3<f64> {
    let (sin_angle, cos_angle) = angle_radians.sin_cos();
    Matrix3::new(
        1.0, 0.0, 0.0, //
        0.0, cos_angle, -sin_angle, //
        0.0, sin_angle, cos_angle,
    )
}

/// Elementary rotation about the Z-axis by `angle_radians`.
pub fn rotation_matrix_about_z_axis(angle_radians: f64) -> Matrix3<f64> {
    let (sin_angle, cos_angle) = angle_radians.sin_cos();
    Matrix3::new(
        cos_angle, -sin_angle, 0.0, //
        sin_angle, cos_angle, 0.0, //
        0.0, 0.0, 1.0,
    )
}

/// Builds the composite 3-1-3 Euler rotation matrix that carries a vector
/// from the perifocal (PQW) frame into the Earth-Centered Inertial (ECI)
/// frame: R = R_z(raan) * R_x(inclination) * R_z(argument_of_perigee).
pub fn perifocal_to_eci_rotation_matrix(
    right_ascension_of_ascending_node_radians: f64,
    inclination_radians: f64,
    argument_of_perigee_radians: f64,
) -> Matrix3<f64> {
    rotation_matrix_about_z_axis(right_ascension_of_ascending_node_radians)
        * rotation_matrix_about_x_axis(inclination_radians)
        * rotation_matrix_about_z_axis(argument_of_perigee_radians)
}

/// Builds the ECI -> ECEF rotation matrix for a given elapsed time since the
/// reference epoch, modeling Earth's rotation as a simple Z-axis spin. This
/// intentionally omits precession/nutation/polar-motion corrections, which
/// are outside the scope of a first-order telemetry processor.
pub fn eci_to_ecef_rotation_matrix(seconds_since_reference_epoch: f64) -> Matrix3<f64> {
    let earth_rotation_angle_radians =
        EARTH_ANGULAR_ROTATION_RATE_RAD_S * seconds_since_reference_epoch;
    rotation_matrix_about_z_axis(earth_rotation_angle_radians).transpose()
}

// ---------------------------------------------------------------------------
// Kepler's equation solver
// ---------------------------------------------------------------------------

/// Solves Kepler's equation M = E - e*sin(E) for the eccentric anomaly E,
/// given the mean anomaly M and eccentricity e, via Newton-Raphson iteration.
pub fn solve_eccentric_anomaly_newton_raphson(
    mean_anomaly_radians: f64,
    eccentricity: f64,
) -> Result<f64, AerospaceError> {
    if !(0.0..1.0).contains(&eccentricity) {
        return Err(AerospaceError::UnsupportedEccentricity { eccentricity });
    }

    let normalized_mean_anomaly =
        mean_anomaly_radians.rem_euclid(2.0 * std::f64::consts::PI);

    let mut eccentric_anomaly_estimate = if eccentricity < 0.8 {
        normalized_mean_anomaly
    } else {
        std::f64::consts::PI
    };

    for iteration_index in 0..KEPLER_SOLVER_MAX_ITERATIONS {
        let kepler_residual = eccentric_anomaly_estimate
            - eccentricity * eccentric_anomaly_estimate.sin()
            - normalized_mean_anomaly;
        let kepler_residual_derivative =
            1.0 - eccentricity * eccentric_anomaly_estimate.cos();

        if kepler_residual_derivative.abs() < f64::EPSILON {
            return Err(AerospaceError::SingularMatrix {
                context: "Kepler equation derivative vanished during Newton-Raphson iteration"
                    .to_string(),
            });
        }

        let newton_step = kepler_residual / kepler_residual_derivative;
        eccentric_anomaly_estimate -= newton_step;

        if newton_step.abs() < KEPLER_SOLVER_TOLERANCE_RADIANS {
            return Ok(eccentric_anomaly_estimate);
        }

        if iteration_index == KEPLER_SOLVER_MAX_ITERATIONS - 1 {
            return Err(AerospaceError::KeplerSolverDivergence {
                iterations_attempted: KEPLER_SOLVER_MAX_ITERATIONS,
                final_residual: kepler_residual,
            });
        }
    }

    Ok(eccentric_anomaly_estimate)
}

// ---------------------------------------------------------------------------
// Keplerian elements <-> Cartesian state vector conversions
// ---------------------------------------------------------------------------

/// Converts classical Keplerian elements into an ECI Cartesian state vector
/// by constructing the position/velocity in the perifocal frame and then
/// rotating them into ECI with the 3-1-3 Euler rotation matrix.
pub fn keplerian_elements_to_state_vector(
    elements: &KeplerianElements,
) -> Result<StateVector, AerospaceError> {
    elements.validate()?;

    let eccentric_anomaly_radians = solve_eccentric_anomaly_newton_raphson(
        elements.mean_anomaly_radians,
        elements.eccentricity,
    )?;

    let (sin_eccentric_anomaly, cos_eccentric_anomaly) = eccentric_anomaly_radians.sin_cos();

    let semi_latus_rectum_meters =
        elements.semi_major_axis_meters * (1.0 - elements.eccentricity.powi(2));

    let orbital_radius_meters =
        elements.semi_major_axis_meters * (1.0 - elements.eccentricity * cos_eccentric_anomaly);

    if orbital_radius_meters <= 0.0 || !orbital_radius_meters.is_finite() {
        return Err(AerospaceError::InvalidOrbitalElements {
            reason: format!(
                "derived orbital radius {orbital_radius_meters} m is non-physical"
            ),
        });
    }

    // Position expressed in the perifocal (PQW) frame:
    //   x_pqw = a * (cos(E) - e)
    //   y_pqw = a * sqrt(1 - e^2) * sin(E)
    // (equivalently x = r*cos(true_anomaly), y = r*sin(true_anomaly); this
    // eccentric-anomaly form avoids a separate true-anomaly computation.)
    let position_perifocal_meters = Vector3::new(
        elements.semi_major_axis_meters * (cos_eccentric_anomaly - elements.eccentricity),
        elements.semi_major_axis_meters
            * (1.0 - elements.eccentricity.powi(2)).sqrt()
            * sin_eccentric_anomaly,
        0.0,
    );

    let mean_motion_rad_s = elements.mean_motion_rad_s();
    let eccentric_anomaly_rate_rad_s =
        mean_motion_rad_s / (1.0 - elements.eccentricity * cos_eccentric_anomaly);

    // Velocity is the time-derivative of the position expressions above:
    //   vx_pqw = -a * sin(E) * dE/dt
    //   vy_pqw =  a * sqrt(1 - e^2) * cos(E) * dE/dt
    let velocity_perifocal_meters_per_second = Vector3::new(
        -elements.semi_major_axis_meters * eccentric_anomaly_rate_rad_s * sin_eccentric_anomaly,
        elements.semi_major_axis_meters
            * eccentric_anomaly_rate_rad_s
            * (1.0 - elements.eccentricity.powi(2)).sqrt()
            * cos_eccentric_anomaly,
        0.0,
    );

    let _ = semi_latus_rectum_meters; // retained for downstream diagnostics/logging

    let rotation_perifocal_to_eci = perifocal_to_eci_rotation_matrix(
        elements.right_ascension_of_ascending_node_radians,
        elements.inclination_radians,
        elements.argument_of_perigee_radians,
    );

    let position_eci_meters = rotation_perifocal_to_eci * position_perifocal_meters;
    let velocity_eci_meters_per_second =
        rotation_perifocal_to_eci * velocity_perifocal_meters_per_second;

    StateVector::new(position_eci_meters, velocity_eci_meters_per_second, elements.epoch)
}

/// Converts an ECI Cartesian state vector into classical Keplerian elements
/// using the standard angular-momentum / eccentricity-vector derivation.
pub fn state_vector_to_keplerian_elements(
    state: &StateVector,
) -> Result<KeplerianElements, AerospaceError> {
    let position_meters = state.position_meters;
    let velocity_meters_per_second = state.velocity_meters_per_second;

    let orbital_radius_meters = position_meters.norm();
    let orbital_speed_meters_per_second = velocity_meters_per_second.norm();

    if orbital_radius_meters < f64::EPSILON {
        return Err(AerospaceError::InvalidStateVector);
    }

    let specific_angular_momentum_vector = position_meters.cross(&velocity_meters_per_second);
    let specific_angular_momentum_magnitude = specific_angular_momentum_vector.norm();

    if specific_angular_momentum_magnitude < f64::EPSILON {
        return Err(AerospaceError::InvalidOrbitalElements {
            reason: "specific angular momentum is degenerate (radial or zero-velocity orbit)"
                .to_string(),
        });
    }

    let node_axis_vector =
        Vector3::new(0.0, 0.0, 1.0).cross(&specific_angular_momentum_vector);
    let node_axis_magnitude = node_axis_vector.norm();

    let mu = EARTH_GRAVITATIONAL_PARAMETER_M3_S2;

    let eccentricity_vector = ((orbital_speed_meters_per_second.powi(2) - mu / orbital_radius_meters)
        * position_meters
        - position_meters.dot(&velocity_meters_per_second) * velocity_meters_per_second)
        / mu;
    let eccentricity = eccentricity_vector.norm();

    let specific_mechanical_energy =
        orbital_speed_meters_per_second.powi(2) / 2.0 - mu / orbital_radius_meters;

    if specific_mechanical_energy >= 0.0 {
        return Err(AerospaceError::InvalidOrbitalElements {
            reason: "specific mechanical energy is non-negative: orbit is parabolic or hyperbolic, not a closed ellipse".to_string(),
        });
    }

    let semi_major_axis_meters = -mu / (2.0 * specific_mechanical_energy);

    let inclination_radians =
        (specific_angular_momentum_vector.z / specific_angular_momentum_magnitude).acos();

    let right_ascension_of_ascending_node_radians = if node_axis_magnitude < f64::EPSILON {
        0.0
    } else {
        let mut raan = (node_axis_vector.x / node_axis_magnitude).acos();
        if node_axis_vector.y < 0.0 {
            raan = 2.0 * std::f64::consts::PI - raan;
        }
        raan
    };

    let argument_of_perigee_radians = if node_axis_magnitude < f64::EPSILON
        || eccentricity < f64::EPSILON
    {
        0.0
    } else {
        let mut argument_of_perigee =
            (node_axis_vector.dot(&eccentricity_vector) / (node_axis_magnitude * eccentricity))
                .clamp(-1.0, 1.0)
                .acos();
        if eccentricity_vector.z < 0.0 {
            argument_of_perigee = 2.0 * std::f64::consts::PI - argument_of_perigee;
        }
        argument_of_perigee
    };

    let true_anomaly_radians = if eccentricity < f64::EPSILON {
        0.0
    } else {
        let mut true_anomaly = (eccentricity_vector.dot(&position_meters)
            / (eccentricity * orbital_radius_meters))
            .clamp(-1.0, 1.0)
            .acos();
        if position_meters.dot(&velocity_meters_per_second) < 0.0 {
            true_anomaly = 2.0 * std::f64::consts::PI - true_anomaly;
        }
        true_anomaly
    };

    let eccentric_anomaly_radians = 2.0
        * ((true_anomaly_radians / 2.0).tan() * ((1.0 - eccentricity) / (1.0 + eccentricity)).sqrt())
            .atan();

    let mean_anomaly_radians = (eccentric_anomaly_radians
        - eccentricity * eccentric_anomaly_radians.sin())
    .rem_euclid(2.0 * std::f64::consts::PI);

    let elements = KeplerianElements {
        semi_major_axis_meters,
        eccentricity,
        inclination_radians,
        right_ascension_of_ascending_node_radians,
        argument_of_perigee_radians,
        mean_anomaly_radians,
        epoch: state.epoch,
    };

    elements.validate()?;
    Ok(elements)
}

// ---------------------------------------------------------------------------
// J2 secular perturbation propagation
// ---------------------------------------------------------------------------

/// Advances a set of Keplerian elements forward by `delta_seconds` under the
/// secular (long-term average) effects of Earth's J2 oblateness: nodal
/// regression, apsidal precession, and the associated mean-motion
/// correction. This is the standard "Simplified Perturbation" analytic
/// update used for coarse propagation between full numerical integration
/// passes.
pub fn propagate_j2_secular_perturbations(
    elements: &KeplerianElements,
    delta_seconds: f64,
) -> Result<KeplerianElements, AerospaceError> {
    elements.validate()?;

    let semi_major_axis_meters = elements.semi_major_axis_meters;
    let eccentricity = elements.eccentricity;
    let inclination_radians = elements.inclination_radians;

    let mean_motion_rad_s = elements.mean_motion_rad_s();

    let semi_latus_rectum_meters = semi_major_axis_meters * (1.0 - eccentricity.powi(2));
    let perturbation_scale_factor = 1.5
        * EARTH_J2_OBLATENESS_COEFFICIENT
        * (EARTH_EQUATORIAL_RADIUS_METERS / semi_latus_rectum_meters).powi(2)
        * mean_motion_rad_s;

    let raan_secular_rate_rad_s = -perturbation_scale_factor * inclination_radians.cos();

    let argument_of_perigee_secular_rate_rad_s = perturbation_scale_factor
        * (2.0 - 2.5 * inclination_radians.sin().powi(2));

    let mean_anomaly_correction_rate_rad_s = perturbation_scale_factor
        * (1.0 - eccentricity.powi(2)).sqrt()
        * (1.0 - 1.5 * inclination_radians.sin().powi(2));

    let updated_raan_radians = (elements.right_ascension_of_ascending_node_radians
        + raan_secular_rate_rad_s * delta_seconds)
        .rem_euclid(2.0 * std::f64::consts::PI);

    let updated_argument_of_perigee_radians = (elements.argument_of_perigee_radians
        + argument_of_perigee_secular_rate_rad_s * delta_seconds)
        .rem_euclid(2.0 * std::f64::consts::PI);

    let updated_mean_anomaly_radians = (elements.mean_anomaly_radians
        + (mean_motion_rad_s + mean_anomaly_correction_rate_rad_s) * delta_seconds)
        .rem_euclid(2.0 * std::f64::consts::PI);

    let updated_epoch = elements.epoch + chrono::Duration::milliseconds((delta_seconds * 1000.0) as i64);

    let updated_elements = KeplerianElements {
        semi_major_axis_meters,
        eccentricity,
        inclination_radians,
        right_ascension_of_ascending_node_radians: updated_raan_radians,
        argument_of_perigee_radians: updated_argument_of_perigee_radians,
        mean_anomaly_radians: updated_mean_anomaly_radians,
        epoch: updated_epoch,
    };

    updated_elements.validate()?;
    Ok(updated_elements)
}

// ---------------------------------------------------------------------------
// Acceleration models (two-body gravity, J2 gravity-gradient, drag)
// ---------------------------------------------------------------------------

/// Newtonian two-body gravitational acceleration: a = -mu * r / |r|^3.
pub fn compute_two_body_gravitational_acceleration(
    position_meters: &Vector3<f64>,
) -> Result<Vector3<f64>, AerospaceError> {
    let orbital_radius_meters = position_meters.norm();
    if orbital_radius_meters < f64::EPSILON {
        return Err(AerospaceError::SingularMatrix {
            context: "position vector norm is degenerate in gravitational acceleration model"
                .to_string(),
        });
    }
    Ok(-EARTH_GRAVITATIONAL_PARAMETER_M3_S2 * position_meters
        / orbital_radius_meters.powi(3))
}

/// J2 oblateness gravity-gradient perturbing acceleration in the ECI frame,
/// derived from the gradient of the J2 term in the geopotential expansion.
pub fn compute_j2_perturbation_acceleration(
    position_meters: &Vector3<f64>,
) -> Result<Vector3<f64>, AerospaceError> {
    let orbital_radius_meters = position_meters.norm();
    if orbital_radius_meters < f64::EPSILON {
        return Err(AerospaceError::SingularMatrix {
            context: "position vector norm is degenerate in J2 acceleration model".to_string(),
        });
    }

    let position_x = position_meters.x;
    let position_y = position_meters.y;
    let position_z = position_meters.z;

    let radius_squared = orbital_radius_meters.powi(2);
    let z_over_r_squared = (position_z * position_z) / radius_squared;

    let common_prefactor = 1.5
        * EARTH_J2_OBLATENESS_COEFFICIENT
        * EARTH_GRAVITATIONAL_PARAMETER_M3_S2
        * EARTH_EQUATORIAL_RADIUS_METERS.powi(2)
        / orbital_radius_meters.powi(5);

    let acceleration_x = common_prefactor * position_x * (5.0 * z_over_r_squared - 1.0);
    let acceleration_y = common_prefactor * position_y * (5.0 * z_over_r_squared - 1.0);
    let acceleration_z = common_prefactor * position_z * (5.0 * z_over_r_squared - 3.0);

    Ok(Vector3::new(acceleration_x, acceleration_y, acceleration_z))
}

/// Atmospheric drag deceleration, computed against the velocity relative to
/// the co-rotating atmosphere (i.e. correcting inertial velocity for Earth's
/// rotation), using the standard drag equation
/// a_drag = -0.5 * rho * (Cd * A / m) * |v_rel| * v_rel.
pub fn compute_atmospheric_drag_acceleration(
    state: &StateVector,
    spacecraft: &SpacecraftPhysicalProperties,
    atmosphere: &AtmosphericModel,
) -> Result<Vector3<f64>, AerospaceError> {
    let altitude_meters = state.altitude_meters();

    if altitude_meters <= 0.0 {
        return Err(AerospaceError::OrbitalDecayBelowSurface { altitude_meters });
    }

    let atmospheric_density_kg_m3 = atmosphere.density_at_altitude(altitude_meters)?;

    let earth_rotation_vector_rad_s = Vector3::new(0.0, 0.0, EARTH_ANGULAR_ROTATION_RATE_RAD_S);
    let co_rotating_atmosphere_velocity =
        earth_rotation_vector_rad_s.cross(&state.position_meters);

    let velocity_relative_to_atmosphere =
        state.velocity_meters_per_second - co_rotating_atmosphere_velocity;
    let relative_speed_meters_per_second = velocity_relative_to_atmosphere.norm();

    let ballistic_area_to_mass_ratio = spacecraft.ballistic_area_to_mass_ratio();

    let drag_acceleration = -0.5
        * atmospheric_density_kg_m3
        * ballistic_area_to_mass_ratio
        * relative_speed_meters_per_second
        * velocity_relative_to_atmosphere;

    if !drag_acceleration.iter().all(|component| component.is_finite()) {
        return Err(AerospaceError::AtmosphericModelOutOfBounds {
            altitude_meters,
            reason: "drag acceleration computation produced a non-finite component".to_string(),
        });
    }

    Ok(drag_acceleration)
}

/// Combined perturbing acceleration model: two-body gravity + J2
/// gravity-gradient + atmospheric drag. This is the derivative function fed
/// into the Runge-Kutta integrator.
fn compute_combined_orbital_acceleration(
    state: &StateVector,
    spacecraft: &SpacecraftPhysicalProperties,
    atmosphere: &AtmosphericModel,
) -> Result<Vector3<f64>, AerospaceError> {
    let two_body_acceleration =
        compute_two_body_gravitational_acceleration(&state.position_meters)?;
    let j2_acceleration = compute_j2_perturbation_acceleration(&state.position_meters)?;
    let drag_acceleration =
        compute_atmospheric_drag_acceleration(state, spacecraft, atmosphere)?;

    Ok(two_body_acceleration + j2_acceleration + drag_acceleration)
}

// ---------------------------------------------------------------------------
// Numerical integration (classical 4th-order Runge-Kutta)
// ---------------------------------------------------------------------------

/// A single derivative evaluation of the orbital ODE system:
/// d(position)/dt = velocity, d(velocity)/dt = acceleration.
struct OrbitalStateDerivative {
    velocity_meters_per_second: Vector3<f64>,
    acceleration_meters_per_second_squared: Vector3<f64>,
}

fn evaluate_orbital_derivative(
    position_meters: &Vector3<f64>,
    velocity_meters_per_second: &Vector3<f64>,
    epoch: DateTime<Utc>,
    spacecraft: &SpacecraftPhysicalProperties,
    atmosphere: &AtmosphericModel,
) -> Result<OrbitalStateDerivative, AerospaceError> {
    let probe_state = StateVector::new(*position_meters, *velocity_meters_per_second, epoch)?;
    let acceleration_meters_per_second_squared =
        compute_combined_orbital_acceleration(&probe_state, spacecraft, atmosphere)?;

    Ok(OrbitalStateDerivative {
        velocity_meters_per_second: *velocity_meters_per_second,
        acceleration_meters_per_second_squared,
    })
}

/// Advances a single state vector forward by `time_step_seconds` using
/// classical 4th-order Runge-Kutta integration of the combined
/// gravity + J2 + drag acceleration model.
pub fn runge_kutta_4_orbital_step(
    current_state: &StateVector,
    spacecraft: &SpacecraftPhysicalProperties,
    atmosphere: &AtmosphericModel,
    time_step_seconds: f64,
) -> Result<StateVector, AerospaceError> {
    if !time_step_seconds.is_finite() || time_step_seconds <= 0.0 {
        return Err(AerospaceError::InvalidIntegrationStep { time_step_seconds });
    }

    let half_step_epoch =
        current_state.epoch + chrono::Duration::milliseconds((time_step_seconds * 500.0) as i64);
    let full_step_epoch =
        current_state.epoch + chrono::Duration::milliseconds((time_step_seconds * 1000.0) as i64);

    let k1 = evaluate_orbital_derivative(
        &current_state.position_meters,
        &current_state.velocity_meters_per_second,
        current_state.epoch,
        spacecraft,
        atmosphere,
    )?;

    let position_after_k1 =
        current_state.position_meters + k1.velocity_meters_per_second * (time_step_seconds / 2.0);
    let velocity_after_k1 = current_state.velocity_meters_per_second
        + k1.acceleration_meters_per_second_squared * (time_step_seconds / 2.0);
    let k2 = evaluate_orbital_derivative(
        &position_after_k1,
        &velocity_after_k1,
        half_step_epoch,
        spacecraft,
        atmosphere,
    )?;

    let position_after_k2 =
        current_state.position_meters + k2.velocity_meters_per_second * (time_step_seconds / 2.0);
    let velocity_after_k2 = current_state.velocity_meters_per_second
        + k2.acceleration_meters_per_second_squared * (time_step_seconds / 2.0);
    let k3 = evaluate_orbital_derivative(
        &position_after_k2,
        &velocity_after_k2,
        half_step_epoch,
        spacecraft,
        atmosphere,
    )?;

    let position_after_k3 =
        current_state.position_meters + k3.velocity_meters_per_second * time_step_seconds;
    let velocity_after_k3 = current_state.velocity_meters_per_second
        + k3.acceleration_meters_per_second_squared * time_step_seconds;
    let k4 = evaluate_orbital_derivative(
        &position_after_k3,
        &velocity_after_k3,
        full_step_epoch,
        spacecraft,
        atmosphere,
    )?;

    let weighted_velocity_sum = k1.velocity_meters_per_second
        + 2.0 * k2.velocity_meters_per_second
        + 2.0 * k3.velocity_meters_per_second
        + k4.velocity_meters_per_second;

    let weighted_acceleration_sum = k1.acceleration_meters_per_second_squared
        + 2.0 * k2.acceleration_meters_per_second_squared
        + 2.0 * k3.acceleration_meters_per_second_squared
        + k4.acceleration_meters_per_second_squared;

    let next_position_meters =
        current_state.position_meters + (time_step_seconds / 6.0) * weighted_velocity_sum;
    let next_velocity_meters_per_second = current_state.velocity_meters_per_second
        + (time_step_seconds / 6.0) * weighted_acceleration_sum;

    StateVector::new(next_position_meters, next_velocity_meters_per_second, full_step_epoch)
}

/// Propagates an orbit forward over `duration_seconds`, sampling the
/// trajectory every `time_step_seconds`, under combined gravity + J2 +
/// atmospheric drag effects. Returns the full sampled trajectory, or an
/// `OrbitalDecayBelowSurface` error the instant the satellite's altitude
/// crosses zero (interpreted as re-entry / structural loss).
pub fn propagate_orbit_with_drag_decay(
    initial_state: &StateVector,
    spacecraft: &SpacecraftPhysicalProperties,
    atmosphere: &AtmosphericModel,
    duration_seconds: f64,
    time_step_seconds: f64,
) -> Result<Vec<StateVector>, AerospaceError> {
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err(AerospaceError::InvalidPropagationDuration { duration_seconds });
    }
    if !time_step_seconds.is_finite() || time_step_seconds <= 0.0 {
        return Err(AerospaceError::InvalidIntegrationStep { time_step_seconds });
    }

    let total_steps = (duration_seconds / time_step_seconds).ceil() as usize;
    let mut trajectory = Vec::with_capacity(total_steps + 1);
    trajectory.push(initial_state.clone());

    let mut current_state = initial_state.clone();
    let mut elapsed_seconds = 0.0_f64;

    while elapsed_seconds < duration_seconds {
        let remaining_seconds = duration_seconds - elapsed_seconds;
        let effective_time_step_seconds = time_step_seconds.min(remaining_seconds);

        current_state = runge_kutta_4_orbital_step(
            &current_state,
            spacecraft,
            atmosphere,
            effective_time_step_seconds,
        )?;

        elapsed_seconds += effective_time_step_seconds;
        trajectory.push(current_state.clone());
    }

    Ok(trajectory)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn reference_epoch() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
    }

    fn reference_circular_leo_elements() -> KeplerianElements {
        KeplerianElements {
            semi_major_axis_meters: EARTH_EQUATORIAL_RADIUS_METERS + 500_000.0,
            eccentricity: 0.001,
            inclination_radians: 51.6_f64.to_radians(),
            right_ascension_of_ascending_node_radians: 30.0_f64.to_radians(),
            argument_of_perigee_radians: 45.0_f64.to_radians(),
            mean_anomaly_radians: 10.0_f64.to_radians(),
            epoch: reference_epoch(),
        }
    }

    #[test]
    fn kepler_solver_converges_for_low_eccentricity() {
        let eccentric_anomaly =
            solve_eccentric_anomaly_newton_raphson(1.0, 0.01).expect("solver should converge");
        assert!(eccentric_anomaly.is_finite());
    }

    #[test]
    fn kepler_solver_rejects_hyperbolic_eccentricity() {
        let result = solve_eccentric_anomaly_newton_raphson(1.0, 1.5);
        assert!(matches!(
            result,
            Err(AerospaceError::UnsupportedEccentricity { .. })
        ));
    }

    #[test]
    fn round_trip_elements_to_state_and_back_is_consistent() {
        let original_elements = reference_circular_leo_elements();
        let state = keplerian_elements_to_state_vector(&original_elements)
            .expect("conversion to state vector should succeed");
        let recovered_elements = state_vector_to_keplerian_elements(&state)
            .expect("conversion back to elements should succeed");

        assert!(
            (original_elements.semi_major_axis_meters - recovered_elements.semi_major_axis_meters)
                .abs()
                < 1.0,
            "semi-major axis should round-trip within 1 meter"
        );
        assert!(
            (original_elements.inclination_radians - recovered_elements.inclination_radians).abs()
                < 1e-9,
            "inclination should round-trip within numerical precision"
        );
    }

    #[test]
    fn circular_orbit_speed_matches_vis_viva_prediction() {
        let elements = reference_circular_leo_elements();
        let state = keplerian_elements_to_state_vector(&elements).unwrap();

        let orbital_radius_meters = state.position_meters.norm();
        let expected_speed_meters_per_second = (EARTH_GRAVITATIONAL_PARAMETER_M3_S2
            * (2.0 / orbital_radius_meters - 1.0 / elements.semi_major_axis_meters))
            .sqrt();

        let actual_speed_meters_per_second = state.velocity_meters_per_second.norm();

        assert!(
            (expected_speed_meters_per_second - actual_speed_meters_per_second).abs() < 1.0,
            "vis-viva speed mismatch: expected {expected_speed_meters_per_second}, got {actual_speed_meters_per_second}"
        );
    }

    #[test]
    fn drag_acceleration_opposes_relative_velocity() {
        let elements = reference_circular_leo_elements();
        let state = keplerian_elements_to_state_vector(&elements).unwrap();
        let spacecraft = SpacecraftPhysicalProperties::new(250.0, 2.2, 3.0).unwrap();
        let atmosphere = AtmosphericModel::low_earth_orbit_default();

        let drag_acceleration =
            compute_atmospheric_drag_acceleration(&state, &spacecraft, &atmosphere).unwrap();

        assert!(drag_acceleration.norm() > 0.0);
        assert!(drag_acceleration.dot(&state.velocity_meters_per_second) < 0.0);
    }

    #[test]
    fn propagation_below_surface_returns_decay_error() {
        let deep_position = Vector3::new(1000.0, 0.0, 0.0);
        let state = StateVector::new(deep_position, Vector3::new(0.0, 7500.0, 0.0), reference_epoch())
            .unwrap();
        let spacecraft = SpacecraftPhysicalProperties::new(250.0, 2.2, 3.0).unwrap();
        let atmosphere = AtmosphericModel::low_earth_orbit_default();

        let result = compute_atmospheric_drag_acceleration(&state, &spacecraft, &atmosphere);
        assert!(matches!(
            result,
            Err(AerospaceError::OrbitalDecayBelowSurface { .. })
        ));
    }

    #[test]
    fn rk4_step_preserves_finite_state() {
        let elements = reference_circular_leo_elements();
        let initial_state = keplerian_elements_to_state_vector(&elements).unwrap();
        let spacecraft = SpacecraftPhysicalProperties::new(250.0, 2.2, 3.0).unwrap();
        let atmosphere = AtmosphericModel::low_earth_orbit_default();

        let next_state =
            runge_kutta_4_orbital_step(&initial_state, &spacecraft, &atmosphere, 10.0).unwrap();

        assert!(next_state.position_meters.iter().all(|c| c.is_finite()));
        assert!(next_state
            .velocity_meters_per_second
            .iter()
            .all(|c| c.is_finite()));
    }

    #[test]
    fn j2_secular_propagation_regresses_node_for_prograde_inclination() {
        let elements = reference_circular_leo_elements();
        let propagated = propagate_j2_secular_perturbations(&elements, 86_400.0).unwrap();

        // For a prograde (i < 90 deg) LEO orbit, J2 causes westward (negative)
        // nodal regression over one day.
        let raan_delta = (propagated.right_ascension_of_ascending_node_radians
            - elements.right_ascension_of_ascending_node_radians)
            .rem_euclid(2.0 * std::f64::consts::PI);
        assert!(raan_delta > std::f64::consts::PI);
    }
}
