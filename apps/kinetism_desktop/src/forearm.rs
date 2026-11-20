//! Forearm pronation/supination ("Forearm Rotation") from the two IMUs.
//!
//! Method:
//! 1. Put the forearm sensor's world frame into the upper sensor's world frame
//!    with the calibrated yaw offset, then take the forearm orientation
//!    relative to the upper arm: `rel = upper^-1 * forearm`. Moving the whole
//!    arm at the shoulder rotates both sensors equally and cancels out.
//! 2. With the forearm in its neutral position (thumb forward, elbow straight)
//!    store `rel_neutral`. Later `delta = rel_neutral^-1 * rel` is the forearm's
//!    rotation since neutral, in the forearm sensor's own frame.
//! 3. Swing-twist decomposition of `delta` about the forearm's distal axis
//!    (from the rest calibration, in sensor coordinates). The twist angle is
//!    the rotation about the forearm's own long axis. Elbow flexion has its
//!    hinge perpendicular to that axis, so it falls into the swing part.
//!
//! This is not wrist flexion, extension or deviation: no sensor sits on the
//! hand, so those are not measured at all.
//!
//! Limits: neither sensor has a magnetometer, so the yaw between the two
//! sensors slowly drifts and shows up as apparent forearm rotation. Capture
//! the neutral position again when the reading at neutral is no longer ~0.

use crate::clinical::Side;
use crate::math::{Quat, Vec3, wrap_angle};

/// Readings are ignored (not scored) when the rest of the forearm's motion
/// (elbow flexion and any other swing) is this large: twist is then ill-defined.
pub const MAX_SWING_DEG: f64 = 160.0;
/// Below this magnitude the forearm counts as neutral.
pub const NEUTRAL_DEADBAND_DEG: f64 = 3.0;
pub const NEUTRAL_SAMPLES: usize = 20;
/// Spread (deg) allowed between samples while the neutral position is held.
pub const NEUTRAL_STILLNESS_DEG: f64 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationCalibration {
    /// Forearm orientation relative to the upper arm at the neutral position.
    neutral_rel: Quat,
    /// Forearm distal axis in the forearm sensor's frame.
    axis: Vec3,
    /// Yaw that maps forearm-sensor world vectors into the upper sensor's world frame.
    yaw_offset: f64,
    /// Where the palm points in the forearm sensor's frame (from the neutral capture).
    palm_sensor: Option<Vec3>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationDirection {
    Pronation,
    Supination,
    Neutral,
}

impl RotationDirection {
    pub fn label(self) -> &'static str {
        match self {
            RotationDirection::Pronation => "Pronation",
            RotationDirection::Supination => "Supination",
            RotationDirection::Neutral => "Neutral",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unreliable {
    /// Neutral position not captured yet.
    NotCalibrated,
    /// No forearm orientation arrived.
    NoForearmSensor,
    /// The forearm's other motion is so large that its twist is ill-defined.
    SwingTooLarge,
    /// The quaternions were not valid unit rotations.
    InvalidData,
}

impl Unreliable {
    pub fn label(self) -> &'static str {
        match self {
            Unreliable::NotCalibrated => "Set the forearm neutral position first",
            Unreliable::NoForearmSensor => "Forearm sensor not reporting",
            Unreliable::SwingTooLarge => "Forearm rotation unreliable at this elbow angle",
            Unreliable::InvalidData => "Invalid sensor data",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForearmRotation {
    /// Pronation positive, supination negative, in (-180, 180].
    pub angle_deg: f64,
    pub direction: RotationDirection,
    /// Rotation of the forearm other than its own twist (mostly elbow flexion).
    pub swing_deg: f64,
}

pub fn direction_of(angle_deg: f64) -> RotationDirection {
    if angle_deg > NEUTRAL_DEADBAND_DEG {
        RotationDirection::Pronation
    } else if angle_deg < -NEUTRAL_DEADBAND_DEG {
        RotationDirection::Supination
    } else {
        RotationDirection::Neutral
    }
}

/// Forearm orientation relative to the upper arm, both in the upper sensor's world frame.
pub fn relative_orientation(upper: Quat, forearm: Quat, yaw_offset: f64) -> Option<Quat> {
    let upper = upper.normalized()?;
    let forearm = forearm.normalized()?;
    let in_upper_world = Quat::from_axis_angle(Vec3::Z, yaw_offset) * forearm;
    (upper.conjugate() * in_upper_world).normalized()
}

/// Swing-twist decomposition of `delta` about `axis` (unit, in the frame
/// `delta` is expressed in): returns `(twist_rad in (-pi, pi], swing_deg)`.
pub fn swing_twist(delta: Quat, axis: Vec3) -> Option<(f64, f64)> {
    let axis = axis.normalized()?;
    let d = delta.normalized()?;
    // q and -q are the same rotation: use the one with w >= 0 so angles stay in range.
    let d = if d.w < 0.0 {
        Quat::new(-d.w, -d.x, -d.y, -d.z)
    } else {
        d
    };
    let along = Vec3::new(d.x, d.y, d.z).dot(axis);
    let Some(twist) = Quat::new(d.w, axis.x * along, axis.y * along, axis.z * along).normalized()
    else {
        // A half-turn about an axis perpendicular to `axis`: all swing, twist undefined.
        return Some((0.0, 180.0));
    };
    let twist_rad = wrap_angle(2.0 * along.atan2(d.w));
    let swing = (d * twist.conjugate()).normalized()?;
    let swing_deg = (2.0 * swing.w.abs().clamp(0.0, 1.0).acos()).to_degrees();
    Some((twist_rad, swing_deg))
}

impl RotationCalibration {
    /// Build from a captured neutral relative orientation.
    pub fn new(neutral_rel: Quat, forearm_axis: Vec3, yaw_offset: f64) -> Option<Self> {
        Some(Self {
            neutral_rel: neutral_rel.normalized()?,
            axis: forearm_axis.normalized()?,
            yaw_offset,
            palm_sensor: None,
        })
    }

    /// Attach the palm direction measured at the neutral capture.
    pub fn with_palm(mut self, palm_sensor: Option<Vec3>) -> Self {
        self.palm_sensor = palm_sensor.and_then(Vec3::normalized);
        self
    }

    /// Palm direction in the forearm sensor's frame.
    pub fn palm_sensor(&self) -> Option<Vec3> {
        self.palm_sensor
    }

    /// Estimate the forearm rotation for one pair of sensor orientations.
    pub fn estimate(
        &self,
        upper: Quat,
        forearm: Option<Quat>,
        side: Side,
    ) -> Result<ForearmRotation, Unreliable> {
        let forearm = forearm.ok_or(Unreliable::NoForearmSensor)?;
        let rel =
            relative_orientation(upper, forearm, self.yaw_offset).ok_or(Unreliable::InvalidData)?;
        let delta = self.neutral_rel.conjugate() * rel;
        let (twist_rad, swing_deg) =
            swing_twist(delta, self.axis).ok_or(Unreliable::InvalidData)?;
        if swing_deg > MAX_SWING_DEG {
            return Err(Unreliable::SwingTooLarge);
        }
        // Positive twist about the distal axis is supination for the right arm.
        let angle_deg = -side.sign() * twist_rad.to_degrees();
        Ok(ForearmRotation {
            angle_deg,
            direction: direction_of(angle_deg),
            swing_deg,
        })
    }
}

/// Collects relative orientations while the neutral position is held still.
#[derive(Clone, Debug, Default)]
pub struct NeutralAccumulator {
    samples: Vec<Quat>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NeutralStatus {
    Collecting(f32),
    /// The forearm moved: collection restarted.
    Moving,
    Done(Quat),
}

impl NeutralAccumulator {
    pub fn progress(&self) -> f32 {
        (self.samples.len() as f32 / NEUTRAL_SAMPLES as f32).min(1.0)
    }

    pub fn push(&mut self, rel: Quat) -> NeutralStatus {
        let Some(rel) = rel.normalized() else {
            self.samples.clear();
            return NeutralStatus::Moving;
        };
        if let Some(first) = self.samples.first() {
            let spread = (first.conjugate() * rel).normalized().map_or(180.0, |d| {
                (2.0 * d.w.abs().clamp(0.0, 1.0).acos()).to_degrees()
            });
            if spread > NEUTRAL_STILLNESS_DEG {
                self.samples.clear();
                self.samples.push(rel);
                return NeutralStatus::Moving;
            }
        }
        self.samples.push(rel);
        if self.samples.len() < NEUTRAL_SAMPLES {
            return NeutralStatus::Collecting(self.progress());
        }
        // Mean of nearby quaternions: align hemispheres, sum, normalise.
        let first = self.samples[0];
        let mut sum = [0.0f64; 4];
        for q in &self.samples {
            let sign = if first.w * q.w + first.x * q.x + first.y * q.y + first.z * q.z < 0.0 {
                -1.0
            } else {
                1.0
            };
            sum[0] += sign * q.w;
            sum[1] += sign * q.x;
            sum[2] += sign * q.y;
            sum[3] += sign * q.z;
        }
        self.samples.clear();
        match Quat::new(sum[0], sum[1], sum[2], sum[3]).normalized() {
            Some(mean) => NeutralStatus::Done(mean),
            None => NeutralStatus::Moving,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::{
        PoseAccumulator, SAMPLES_PER_POSE, SimArm, calibration_without_forward_pose,
        capture_forward, capture_rest,
    };

    /// Calibrate a simulated arm and capture its neutral forearm position.
    fn calibrated(side: Side) -> (SimArm, RotationCalibration) {
        let mut sim = SimArm::new(side);
        let mut acc = PoseAccumulator::default();
        for _ in 0..SAMPLES_PER_POSE {
            let (u, f) = sim.quats();
            acc.push(u, Some(f));
        }
        let rest = capture_rest(&acc).expect("rest");
        acc.clear();
        sim.elevation_deg = 60.0;
        sim.plane_deg = 90.0;
        for _ in 0..SAMPLES_PER_POSE {
            let (u, f) = sim.quats();
            acc.push(u, Some(f));
        }
        let cal = capture_forward(&rest, &acc).expect("forward");
        let (u, f) = sim.quats();
        let rel = relative_orientation(u, f, cal.forearm_yaw_offset).unwrap();
        let rot = RotationCalibration::new(rel, cal.forearm_axis.unwrap(), cal.forearm_yaw_offset)
            .unwrap();
        (sim, rot)
    }

    fn estimate(sim: &SimArm, rot: &RotationCalibration) -> Result<ForearmRotation, Unreliable> {
        let (u, f) = sim.quats();
        rot.estimate(u, Some(f), sim.side)
    }

    #[test]
    fn neutral_reads_zero_and_neutral() {
        for side in [Side::Right, Side::Left] {
            let (sim, rot) = calibrated(side);
            let r = estimate(&sim, &rot).unwrap();
            assert!(r.angle_deg.abs() < 0.5, "{side:?}: {}", r.angle_deg);
            assert_eq!(r.direction, RotationDirection::Neutral);
        }
    }

    #[test]
    fn recovers_pronation_and_supination_for_both_arms() {
        for side in [Side::Right, Side::Left] {
            let (mut sim, rot) = calibrated(side);
            for truth in [-80.0, -45.0, -10.0, 10.0, 30.0, 60.0, 85.0] {
                sim.pronation_deg = truth;
                let r = estimate(&sim, &rot).unwrap();
                assert!(
                    (r.angle_deg - truth).abs() < 0.5,
                    "{side:?}: {truth} read as {}",
                    r.angle_deg
                );
                assert_eq!(
                    r.direction,
                    if truth > 0.0 {
                        RotationDirection::Pronation
                    } else {
                        RotationDirection::Supination
                    },
                    "{side:?} {truth}"
                );
            }
        }
    }

    #[test]
    fn rotation_is_independent_of_shoulder_position_and_elbow_flexion() {
        for side in [Side::Right, Side::Left] {
            let (mut sim, rot) = calibrated(side);
            sim.pronation_deg = 40.0;
            for (e, p, el) in [
                (0.0, 0.0, 0.0),
                (90.0, 0.0, 0.0),
                (150.0, -20.0, 0.0),
                (45.0, 70.0, 60.0),
                (120.0, 30.0, 110.0),
            ] {
                sim.elevation_deg = e;
                sim.plane_deg = p;
                sim.elbow_deg = el;
                let r = estimate(&sim, &rot).unwrap();
                assert!(
                    (r.angle_deg - 40.0).abs() < 0.5,
                    "{side:?} e={e} p={p} elbow={el}: {}",
                    r.angle_deg
                );
            }
        }
    }

    #[test]
    fn rotation_wraps_instead_of_jumping_past_180_degrees() {
        let (mut sim, rot) = calibrated(Side::Right);
        sim.pronation_deg = 170.0;
        let a = estimate(&sim, &rot).unwrap().angle_deg;
        sim.pronation_deg = 190.0;
        let b = estimate(&sim, &rot).unwrap().angle_deg;
        assert!((a - 170.0).abs() < 0.5);
        assert!((b + 170.0).abs() < 0.5, "190 deg wraps to -170: {b}");
    }

    #[test]
    fn unreliable_inputs_are_refused_not_guessed() {
        let (sim, rot) = calibrated(Side::Right);
        let (u, f) = sim.quats();
        assert_eq!(
            rot.estimate(u, None, Side::Right),
            Err(Unreliable::NoForearmSensor)
        );
        assert_eq!(
            rot.estimate(Quat::new(0.0, 0.0, 0.0, 0.0), Some(f), Side::Right),
            Err(Unreliable::InvalidData)
        );
        assert_eq!(
            rot.estimate(u, Some(Quat::new(f64::NAN, 0.0, 0.0, 1.0)), Side::Right),
            Err(Unreliable::InvalidData)
        );
        // Forearm flipped 180 deg about an axis perpendicular to its own axis.
        let flipped = f * Quat::from_axis_angle(Vec3::new(0.3, 0.0, 1.0), std::f64::consts::PI);
        let r = rot.estimate(u, Some(flipped), Side::Right);
        assert!(
            matches!(r, Err(Unreliable::SwingTooLarge)) || r.is_ok_and(|r| r.swing_deg > 90.0),
            "{r:?}"
        );
    }

    #[test]
    fn swing_twist_separates_the_two_parts() {
        let axis = Vec3::new(0.0, 0.0, -1.0);
        let swing = Quat::from_axis_angle(Vec3::X, 1.0);
        let twist = Quat::from_axis_angle(axis, 0.6);
        let (t, s) = swing_twist(swing * twist, axis).unwrap();
        assert!((t - 0.6).abs() < 1e-9, "{t}");
        assert!((s - 1.0_f64.to_degrees()).abs() < 1e-6, "{s}");
        // q and -q are the same rotation.
        let q = swing * twist;
        let neg = Quat::new(-q.w, -q.x, -q.y, -q.z);
        assert!((swing_twist(neg, axis).unwrap().0 - 0.6).abs() < 1e-9);
        assert_eq!(swing_twist(Quat::new(0.0, 0.0, 0.0, 0.0), axis), None);
        assert_eq!(swing_twist(Quat::IDENTITY, Vec3::ZERO), None);
    }

    #[test]
    fn direction_uses_a_small_neutral_deadband() {
        assert_eq!(direction_of(0.0), RotationDirection::Neutral);
        assert_eq!(
            direction_of(NEUTRAL_DEADBAND_DEG),
            RotationDirection::Neutral
        );
        assert_eq!(direction_of(3.5), RotationDirection::Pronation);
        assert_eq!(direction_of(-3.5), RotationDirection::Supination);
    }

    #[test]
    fn neutral_capture_needs_stillness_and_averages_the_samples() {
        let base = Quat::from_axis_angle(Vec3::Y, 0.4);
        let mut acc = NeutralAccumulator::default();
        for i in 0..NEUTRAL_SAMPLES - 1 {
            let jitter = Quat::from_axis_angle(Vec3::X, 0.002 * (i % 3) as f64);
            assert!(matches!(
                acc.push(base * jitter),
                NeutralStatus::Collecting(_)
            ));
        }
        let NeutralStatus::Done(mean) = acc.push(base) else {
            panic!("done after enough samples");
        };
        assert!((base.conjugate() * mean).w.abs() > 0.9999);

        // Moving restarts the collection.
        let mut acc = NeutralAccumulator::default();
        assert!(matches!(acc.push(base), NeutralStatus::Collecting(_)));
        let far = base * Quat::from_axis_angle(Vec3::X, 0.5);
        assert_eq!(acc.push(far), NeutralStatus::Moving);
        assert_eq!(
            acc.push(Quat::new(0.0, 0.0, 0.0, 0.0)),
            NeutralStatus::Moving
        );
        assert!(
            acc.progress() < 0.01,
            "an invalid sample clears the capture"
        );
    }

    #[test]
    fn calibration_rejects_invalid_inputs() {
        assert!(RotationCalibration::new(Quat::new(0.0, 0.0, 0.0, 0.0), Vec3::X, 0.0).is_none());
        assert!(RotationCalibration::new(Quat::IDENTITY, Vec3::ZERO, 0.0).is_none());
    }

    #[test]
    fn rest_only_calibration_still_gives_rotation() {
        // Skipping the forward pose leaves the sensors' relative yaw unknown,
        // so the neutral capture (which uses the same offset) must still read ~0 at neutral.
        let mut sim = SimArm::new(Side::Right);
        let mut acc = PoseAccumulator::default();
        for _ in 0..SAMPLES_PER_POSE {
            let (u, f) = sim.quats();
            acc.push(u, Some(f));
        }
        let rest = capture_rest(&acc).unwrap();
        let cal = calibration_without_forward_pose(&rest, Side::Right);
        let (u, f) = sim.quats();
        let rel = relative_orientation(u, f, cal.forearm_yaw_offset).unwrap();
        let rot = RotationCalibration::new(rel, cal.forearm_axis.unwrap(), cal.forearm_yaw_offset)
            .unwrap();
        assert!(estimate(&sim, &rot).unwrap().angle_deg.abs() < 0.5);
        sim.pronation_deg = 50.0;
        // The yaw offset is wrong here, but the shoulder is at rest, so twist is unaffected.
        let angle = estimate(&sim, &rot).unwrap().angle_deg;
        assert!((angle - 50.0).abs() < 0.5, "{angle}");
    }
}
