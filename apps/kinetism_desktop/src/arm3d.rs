//! 3D arm model: forward kinematics and the orbit camera used by the 3D view.
//!
//! Everything here is derived from segment *orientations*. No IMU position is
//! integrated, so absolute hand position is never measured: the wrist is
//! placed by chaining the two segment directions from a fixed shoulder, using
//! the same fixed segment-length split as the 2D game ([`UPPER_SHARE`]).
//!
//! Body frame (after calibration): x = patient's right, y = forward, z = up.
//! Units are arm lengths (acromion to wrist = 1.0).
//!
//! Known limits, shown in the UI as well:
//! - The shoulder is fixed at the origin. There is no trunk sensor, so trunk
//!   lean or shoulder-girdle movement is not visible.
//! - There is no hand sensor. The hand and blade are rigidly aligned with the
//!   forearm, so their orientation is inferred, not measured wrist articulation.

use crate::clinical::Side;
use crate::kinematics::UPPER_SHARE;
use crate::math::{Quat, Vec3};

/// Length of the drawn hand beyond the wrist (arm lengths). Inferred, not measured.
pub const HAND_LENGTH: f64 = 0.12;
/// Length of the virtual blade beyond the hand (arm lengths).
pub const BLADE_LENGTH: f64 = 0.30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmSkeleton {
    pub shoulder: Vec3,
    pub elbow: Vec3,
    pub wrist: Vec3,
    /// End of the drawn hand (inferred: continues the forearm axis).
    pub hand_tip: Vec3,
    /// End of the virtual blade.
    pub blade_tip: Vec3,
    pub upper_dir: Vec3,
    pub forearm_dir: Vec3,
    /// Unit vector out of the palm side of the hand. Display only: it is the
    /// neutral palm direction turned by the estimated forearm rotation.
    pub palm_normal: Vec3,
    /// Unit vector along the thumb side of the hand (perpendicular to the
    /// forearm and the palm normal). Display only, like `palm_normal`.
    pub thumb_dir: Vec3,
    /// `false` when no forearm sensor reports: the forearm then simply
    /// continues the upper arm and must not be read as a measurement.
    pub forearm_measured: bool,
}

/// Neutral (mid-position, thumb forward) palm direction for a forearm pointing
/// along `forearm_dir`: medial, projected perpendicular to the forearm.
pub(crate) fn neutral_palm_normal(forearm_dir: Vec3, side: Side) -> Vec3 {
    let medial = Vec3::X.scale(-side.sign());
    for reference in [medial, Vec3::Y] {
        let perpendicular = reference - forearm_dir.scale(reference.dot(forearm_dir));
        if let Some(n) = perpendicular.normalized() {
            return n;
        }
    }
    Vec3::X
}

/// Chain the two segments from the shoulder.
///
/// `forearm_dir` is `None` when the forearm sensor is missing.
/// `pronation_deg` is the estimated forearm rotation (pronation positive,
/// supination negative) or `None` when it is unknown or unreliable.
pub fn skeleton(
    upper_dir: Vec3,
    forearm_dir: Option<Vec3>,
    pronation_deg: Option<f64>,
    side: Side,
) -> ArmSkeleton {
    let upper = upper_dir.normalized().unwrap_or(Vec3::DOWN);
    let forearm_measured = forearm_dir.is_some();
    let forearm = forearm_dir.and_then(Vec3::normalized).unwrap_or(upper);
    let shoulder = Vec3::ZERO;
    let elbow = upper.scale(UPPER_SHARE);
    let wrist = elbow + forearm.scale(1.0 - UPPER_SHARE);
    let hand_tip = wrist + forearm.scale(HAND_LENGTH);
    let blade_tip = hand_tip + forearm.scale(BLADE_LENGTH);
    let neutral = neutral_palm_normal(forearm, side);
    // Pronation is a rotation about the distal axis, with the sign flipped for the right arm.
    let pronation = pronation_deg.filter(|d| d.is_finite()).unwrap_or(0.0);
    let twist = Quat::from_axis_angle(forearm, -side.sign() * pronation.to_radians());
    let palm_normal = twist.rotate(neutral);
    // Mid-position, hanging arm: palm medial, thumb forward. Mirrored per arm.
    let thumb_dir = forearm
        .cross(palm_normal)
        .scale(side.sign())
        .normalized()
        .unwrap_or(Vec3::Y);
    ArmSkeleton {
        shoulder,
        elbow,
        wrist,
        hand_tip,
        blade_tip,
        upper_dir: upper,
        forearm_dir: forearm,
        palm_normal,
        thumb_dir,
        forearm_measured,
    }
}

/// Like [`skeleton`], but the palm direction comes from the forearm's full
/// orientation (shoulder rotation, elbow and forearm roll together) instead of
/// being re-derived from the forearm direction and a rotation angle.
/// `palm_hint` is projected perpendicular to the forearm.
pub fn skeleton_with_hand(
    upper_dir: Vec3,
    forearm_dir: Option<Vec3>,
    palm_hint: Vec3,
    side: Side,
) -> ArmSkeleton {
    let mut sk = skeleton(upper_dir, forearm_dir, None, side);
    let f = sk.forearm_dir;
    if let Some(n) = (palm_hint - f.scale(palm_hint.dot(f))).normalized() {
        sk.palm_normal = n;
        sk.thumb_dir = f
            .cross(n)
            .scale(side.sign())
            .normalized()
            .unwrap_or(sk.thumb_dir);
    }
    sk
}

/// The reference pose shown beside the live arm: arm hanging, elbow straight.
pub fn neutral_skeleton(side: Side) -> ArmSkeleton {
    skeleton(Vec3::DOWN, Some(Vec3::DOWN), Some(0.0), side)
}

/// Distance from `point` to the segment `a`-`b`.
pub fn segment_point_distance(a: Vec3, b: Vec3, point: Vec3) -> f64 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 < 1e-12 {
        return (point - a).length();
    }
    let t = ((point - a).dot(ab) / len2).clamp(0.0, 1.0);
    (point - (a + ab.scale(t))).length()
}

/// Orbit camera around a target point. At `yaw = 0, pitch = 0` it looks from
/// behind the patient, so the patient's right is screen right (the same
/// mirror convention as the 2D game).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub target: Vec3,
    pub fov_deg: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projected {
    /// Pixels from the viewport centre; x right, y down.
    pub x: f64,
    pub y: f64,
    /// Distance along the view direction (arm lengths); larger is farther.
    pub depth: f64,
}

pub const MIN_DISTANCE: f64 = 1.4;
pub const MAX_DISTANCE: f64 = 8.0;
const MAX_PITCH: f64 = 1.45;
const NEAR_PLANE: f64 = 0.05;

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            yaw: 0.45,
            pitch: 0.25,
            distance: 3.6,
            target: Vec3::new(0.0, 0.25, -0.2),
            fov_deg: 40.0,
        }
    }
}

impl OrbitCamera {
    /// Straight-behind view: the 2D game's orientation.
    pub fn behind() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            ..Self::default()
        }
    }

    pub fn orbit(&mut self, d_yaw: f64, d_pitch: f64) {
        if d_yaw.is_finite() {
            self.yaw = crate::math::wrap_angle(self.yaw + d_yaw);
        }
        if d_pitch.is_finite() {
            self.pitch = (self.pitch + d_pitch).clamp(-MAX_PITCH, MAX_PITCH);
        }
    }

    /// Multiplicative zoom: positive `amount` moves closer.
    pub fn zoom(&mut self, amount: f64) {
        if amount.is_finite() {
            self.distance = (self.distance * (-amount).exp()).clamp(MIN_DISTANCE, MAX_DISTANCE);
        }
    }

    fn direction_to_camera(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            -self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
        )
    }

    pub fn position(&self) -> Vec3 {
        self.target + self.direction_to_camera().scale(self.distance)
    }

    /// Camera basis `(right, up, forward)`.
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = -self.direction_to_camera();
        let right = forward.cross(Vec3::Z).normalized().unwrap_or(Vec3::X);
        let up = right.cross(forward);
        (right, up, forward)
    }

    /// Perspective projection into a viewport of the given pixel height. `None`
    /// when the point is behind the near plane.
    pub fn project(&self, p: Vec3, viewport_height: f64) -> Option<Projected> {
        let (right, up, forward) = self.basis();
        let v = p - self.position();
        let depth = v.dot(forward);
        if depth < NEAR_PLANE || !depth.is_finite() {
            return None;
        }
        let focal = (viewport_height * 0.5) / (self.fov_deg.to_radians() * 0.5).tan();
        Some(Projected {
            x: v.dot(right) / depth * focal,
            y: -v.dot(up) / depth * focal,
            depth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-9
    }

    #[test]
    fn hanging_arm_points_straight_down_at_full_length() {
        let s = neutral_skeleton(Side::Right);
        assert!(close(s.elbow, Vec3::new(0.0, 0.0, -UPPER_SHARE)));
        assert!(close(s.wrist, Vec3::new(0.0, 0.0, -1.0)));
        assert!(close(s.hand_tip, Vec3::new(0.0, 0.0, -1.0 - HAND_LENGTH)));
        assert!(
            close(
                s.blade_tip,
                Vec3::new(0.0, 0.0, -1.0 - HAND_LENGTH - BLADE_LENGTH)
            ),
            "the blade continues the forearm"
        );
        assert!(s.forearm_measured);
    }

    #[test]
    fn wrist_follows_the_chained_segment_directions() {
        let upper = Vec3::new(1.0, 0.0, 0.0);
        let forearm = Vec3::new(0.0, 1.0, 0.0);
        let s = skeleton(upper, Some(forearm), None, Side::Right);
        assert!(close(s.elbow, Vec3::new(UPPER_SHARE, 0.0, 0.0)));
        assert!(close(
            s.wrist,
            Vec3::new(UPPER_SHARE, 1.0 - UPPER_SHARE, 0.0)
        ));
        let reach = s.wrist.length();
        assert!(reach < 1.0, "a bent elbow shortens the reach: {reach}");
        let straight = skeleton(upper, Some(upper), None, Side::Right);
        assert!((straight.wrist.length() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn segment_lengths_stay_fixed_for_any_direction() {
        for (u, f) in [
            (Vec3::new(0.3, 0.8, -0.2), Vec3::new(-0.6, 0.1, 0.9)),
            (Vec3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -7.0)),
        ] {
            let s = skeleton(u, Some(f), Some(20.0), Side::Left);
            assert!(((s.elbow - s.shoulder).length() - UPPER_SHARE).abs() < 1e-9);
            assert!(((s.wrist - s.elbow).length() - (1.0 - UPPER_SHARE)).abs() < 1e-9);
            assert!(((s.hand_tip - s.wrist).length() - HAND_LENGTH).abs() < 1e-9);
            assert!(((s.blade_tip - s.hand_tip).length() - BLADE_LENGTH).abs() < 1e-9);
        }
    }

    #[test]
    fn missing_forearm_sensor_continues_the_upper_arm_and_says_so() {
        let s = skeleton(Vec3::X, None, None, Side::Right);
        assert!(!s.forearm_measured);
        assert!(close(s.wrist, Vec3::X), "arm drawn straight");
    }

    #[test]
    fn degenerate_directions_never_produce_nan() {
        let s = skeleton(Vec3::ZERO, Some(Vec3::ZERO), Some(f64::NAN), Side::Right);
        for p in [s.elbow, s.wrist, s.hand_tip, s.blade_tip, s.palm_normal] {
            assert!(
                p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
                "{p:?}"
            );
        }
    }

    #[test]
    fn pronation_turns_the_palm_in_mirrored_directions_for_each_arm() {
        // Hanging arm: neutral palm faces the body, pronation turns it backward
        // (thumb moves medially), supination turns it forward (palm "up" / anatomical).
        for side in [Side::Right, Side::Left] {
            let neutral = skeleton(Vec3::DOWN, Some(Vec3::DOWN), Some(0.0), side).palm_normal;
            assert!(close(neutral, Vec3::X.scale(-side.sign())), "{side:?}");
            let pronated = skeleton(Vec3::DOWN, Some(Vec3::DOWN), Some(90.0), side).palm_normal;
            let supinated = skeleton(Vec3::DOWN, Some(Vec3::DOWN), Some(-90.0), side).palm_normal;
            assert!(close(pronated, -supinated), "{side:?}: opposite turns");
            assert!(
                pronated.y.abs() > 0.99,
                "{side:?}: palm turns fore/aft: {pronated:?}"
            );
            assert!(
                pronated.y < 0.0,
                "{side:?}: pronation turns the palm backward on a hanging arm: {pronated:?}"
            );
        }
    }

    #[test]
    fn thumb_points_forward_at_neutral_and_turns_medially_with_pronation() {
        for side in [Side::Right, Side::Left] {
            let hang = |p| skeleton(Vec3::DOWN, Some(Vec3::DOWN), Some(p), side);
            let n = hang(0.0);
            assert!(close(n.thumb_dir, Vec3::Y), "{side:?}: {:?}", n.thumb_dir);
            let pronated = hang(90.0);
            assert!(
                close(pronated.thumb_dir, Vec3::X.scale(-side.sign())),
                "{side:?}: thumb moves toward the body: {:?}",
                pronated.thumb_dir
            );
            for p in [-90.0, 0.0, 35.0, 90.0] {
                let s = hang(p);
                assert!(s.thumb_dir.dot(s.forearm_dir).abs() < 1e-9);
                assert!(s.thumb_dir.dot(s.palm_normal).abs() < 1e-9);
                assert!((s.thumb_dir.length() - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn a_given_palm_direction_is_made_perpendicular_to_the_forearm_and_sets_the_thumb() {
        for side in [Side::Right, Side::Left] {
            let sk = skeleton_with_hand(Vec3::X, Some(Vec3::Y), Vec3::new(0.3, 0.9, -0.5), side);
            assert!(sk.palm_normal.dot(sk.forearm_dir).abs() < 1e-9);
            assert!((sk.palm_normal.length() - 1.0).abs() < 1e-9);
            assert!(sk.thumb_dir.dot(sk.palm_normal).abs() < 1e-9);
            assert!(sk.thumb_dir.dot(sk.forearm_dir).abs() < 1e-9);
            // A useless hint (parallel to the forearm) keeps the neutral hand instead of NaN.
            let kept = skeleton_with_hand(Vec3::X, Some(Vec3::Y), Vec3::Y, side);
            assert!(kept.palm_normal.x.is_finite());
            assert!(close(
                kept.palm_normal,
                skeleton(Vec3::X, Some(Vec3::Y), None, side).palm_normal
            ));
        }
    }

    #[test]
    fn pronation_by_180_flips_the_palm() {
        let a = skeleton(Vec3::X, Some(Vec3::X), Some(0.0), Side::Right).palm_normal;
        let b = skeleton(Vec3::X, Some(Vec3::X), Some(180.0), Side::Right).palm_normal;
        assert!(close(a, -b));
    }

    #[test]
    fn segment_distance_clamps_to_the_end_points() {
        let a = Vec3::ZERO;
        let b = Vec3::new(2.0, 0.0, 0.0);
        assert!((segment_point_distance(a, b, Vec3::new(1.0, 1.0, 0.0)) - 1.0).abs() < 1e-12);
        assert!((segment_point_distance(a, b, Vec3::new(3.0, 0.0, 0.0)) - 1.0).abs() < 1e-12);
        assert!((segment_point_distance(a, a, Vec3::new(0.0, 3.0, 0.0)) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn default_camera_sees_the_patients_right_on_the_screens_right() {
        let cam = OrbitCamera::behind();
        let right = cam
            .project(Vec3::new(0.5, 0.0, 0.0) + cam.target, 600.0)
            .unwrap();
        let left = cam
            .project(Vec3::new(-0.5, 0.0, 0.0) + cam.target, 600.0)
            .unwrap();
        assert!(right.x > 0.0 && left.x < 0.0, "{right:?} {left:?}");
        let up = cam
            .project(Vec3::new(0.0, 0.0, 0.5) + cam.target, 600.0)
            .unwrap();
        assert!(up.y < 0.0, "up is toward the top of the screen");
        let centre = cam.project(cam.target, 600.0).unwrap();
        assert!(centre.x.abs() < 1e-9 && centre.y.abs() < 1e-9);
    }

    #[test]
    fn nearer_points_appear_larger_and_points_behind_the_camera_are_dropped() {
        let cam = OrbitCamera::behind();
        let near = cam
            .project(cam.target + Vec3::new(0.5, -1.0, 0.0), 600.0)
            .unwrap();
        let far = cam
            .project(cam.target + Vec3::new(0.5, 1.0, 0.0), 600.0)
            .unwrap();
        assert!(near.x > far.x && near.depth < far.depth);
        assert!(
            cam.project(cam.position() - Vec3::new(0.0, 1.0, 0.0), 600.0)
                .is_none()
        );
    }

    #[test]
    fn orbit_wraps_yaw_clamps_pitch_and_ignores_bad_input() {
        let mut cam = OrbitCamera::default();
        cam.orbit(10.0, 10.0);
        assert!(cam.yaw.abs() <= std::f64::consts::PI);
        assert!((cam.pitch - 1.45).abs() < 1e-12);
        let before = cam;
        cam.orbit(f64::NAN, f64::INFINITY);
        assert_eq!(cam.yaw, before.yaw);
        cam.orbit(0.0, -100.0);
        assert!((cam.pitch + 1.45).abs() < 1e-12);
    }

    #[test]
    fn zoom_is_bounded_and_the_camera_orbits_at_a_fixed_distance() {
        let mut cam = OrbitCamera::default();
        for _ in 0..50 {
            cam.zoom(1.0);
        }
        assert!((cam.distance - MIN_DISTANCE).abs() < 1e-12);
        for _ in 0..50 {
            cam.zoom(-1.0);
        }
        assert!((cam.distance - MAX_DISTANCE).abs() < 1e-12);
        cam.zoom(f64::NAN);
        assert!((cam.distance - MAX_DISTANCE).abs() < 1e-12);
        cam.orbit(1.1, 0.4);
        assert!(((cam.position() - cam.target).length() - cam.distance).abs() < 1e-9);
    }
}
