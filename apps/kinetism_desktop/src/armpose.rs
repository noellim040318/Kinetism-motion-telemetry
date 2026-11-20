//! Joint-space arm model: shoulder (3 degrees of freedom), elbow hinge and
//! forearm rotation, with forward kinematics. One chain, shared by the
//! simulator, the PNF patterns, the game and the 3D view.
//!
//! Body frame: x = patient's right, y = forward, z = up. Arm lengths are the
//! patient's (acromion-wrist = 1.0, upper arm [`UPPER_SHARE`]); segment
//! lengths never change.
//!
//! Hierarchy: shoulder -> humerus -> elbow (hinge) -> forearm -> wrist -> hand.
//!
//! Shoulder, as in the ISB plane-of-elevation description, applied in order:
//! 1. *plane of elevation* `plane_deg` (0 = abduction, 90 = flexion, 180 =
//!    horizontal adduction across the body, negative = behind the body),
//! 2. *elevation* `elevation_deg` (0 = arm hanging, 90 = horizontal, 180 =
//!    overhead) about the horizontal axis perpendicular to that plane,
//! 3. *axial rotation* `axial_deg` about the humerus's own long axis (external
//!    rotation positive).
//!
//! Elbow: a hinge about the humerus's lateral axis. In anatomical position
//! flexion brings the forearm forward, and the hinge axis then turns with the
//! humerus, so the bending plane always follows the shoulder's axial rotation
//! instead of being invented per pose.
//!
//! Forearm rotation (pronation positive) turns the forearm about its own long
//! axis. It moves the palm only; it cannot move the wrist.
//!
//! Hardware note: two IMUs report segment orientations, nothing else. This
//! model is driven either by those measurements (through the calibration) or
//! by the simulator. Measured data is never "corrected" to fit this model.

use crate::arm3d::{ArmSkeleton, skeleton_with_hand};
use crate::clinical::Side;
use crate::math::{Quat, Vec3};

/// Anatomical shoulder/elbow/forearm angles. The same numbers describe the
/// mirrored pose on the left arm: side-dependent signs are built in below.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointPose {
    pub plane_deg: f64,
    pub elevation_deg: f64,
    /// Humeral rotation about its long axis, external positive.
    pub axial_deg: f64,
    /// 0 = straight elbow.
    pub elbow_deg: f64,
    /// Pronation positive, supination negative.
    pub pronation_deg: f64,
}

impl JointPose {
    pub const REST: JointPose = JointPose {
        plane_deg: 0.0,
        elevation_deg: 0.0,
        axial_deg: 0.0,
        elbow_deg: 0.0,
        pronation_deg: 0.0,
    };

    /// Humerus orientation: segment frame (long axis = down) to body frame.
    pub fn humerus(&self, side: Side) -> Quat {
        let plane = self.plane_deg.to_radians();
        let direction = Vec3::new(side.sign() * plane.cos(), plane.sin(), 0.0);
        // Elevation turns the down-pointing arm toward `direction`.
        let swing =
            Quat::from_axis_angle(Vec3::DOWN.cross(direction), self.elevation_deg.to_radians());
        // External rotation turns the front of the arm outward: positive about
        // "down" for the right arm, mirrored for the left.
        let axial = Quat::from_axis_angle(Vec3::DOWN, side.sign() * self.axial_deg.to_radians());
        swing * axial
    }

    pub fn state(&self, side: Side) -> ArmState {
        ArmState {
            humerus: self.humerus(side),
            elbow_deg: self.elbow_deg,
            pronation_deg: self.pronation_deg,
        }
    }
}

/// Plausible ranges for simulated poses. Real sensor data is never clamped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointLimits {
    pub elevation: (f64, f64),
    pub axial: (f64, f64),
    pub elbow: (f64, f64),
    pub pronation: (f64, f64),
}

impl JointLimits {
    pub const DEFAULT: JointLimits = JointLimits {
        elevation: (0.0, 180.0),
        axial: (-90.0, 90.0),
        // No hyperextension, and no flexion beyond what the arm can do.
        elbow: (0.0, 145.0),
        pronation: (-90.0, 90.0),
    };

    pub fn clamp(&self, pose: JointPose) -> JointPose {
        let fix = |v: f64, (lo, hi): (f64, f64), fallback: f64| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        };
        JointPose {
            plane_deg: if pose.plane_deg.is_finite() {
                pose.plane_deg
            } else {
                0.0
            },
            elevation_deg: fix(pose.elevation_deg, self.elevation, 0.0),
            axial_deg: fix(pose.axial_deg, self.axial, 0.0),
            elbow_deg: fix(pose.elbow_deg, self.elbow, 0.0),
            pronation_deg: fix(pose.pronation_deg, self.pronation, 0.0),
        }
    }
}

/// The arm in the form forward kinematics needs: a shoulder orientation that
/// can be interpolated by quaternion slerp, plus elbow and forearm angles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmState {
    pub humerus: Quat,
    pub elbow_deg: f64,
    pub pronation_deg: f64,
}

impl ArmState {
    /// Elbow flexion axis in the body frame.
    #[cfg(test)]
    pub fn hinge_axis(&self) -> Vec3 {
        self.humerus.rotate(Vec3::X)
    }

    /// Orientation of the forearm segment before its own rotation.
    fn forearm_swing(&self) -> Quat {
        self.humerus * Quat::from_axis_angle(Vec3::X, self.elbow_deg.to_radians())
    }

    /// Forearm orientation including pronation/supination.
    pub fn forearm_orientation(&self, side: Side) -> Quat {
        self.forearm_swing()
            * Quat::from_axis_angle(Vec3::DOWN, -side.sign() * self.pronation_deg.to_radians())
    }

    /// Long-axis directions `(upper arm, forearm)` pointing away from the shoulder.
    pub fn directions(&self) -> (Vec3, Vec3) {
        (
            self.humerus.rotate(Vec3::DOWN),
            self.forearm_swing().rotate(Vec3::DOWN),
        )
    }

    /// The arm in the body frame. The palm comes from the forearm's full
    /// orientation (humerus, elbow and forearm roll), so humeral rotation and
    /// the elbow's bending plane turn the hand too, not only pronation.
    pub fn skeleton(&self, side: Side) -> ArmSkeleton {
        let (upper, forearm) = self.directions();
        // Hanging arm, mid-position: the palm faces the body (medial).
        let palm = self
            .forearm_orientation(side)
            .rotate(Vec3::X.scale(-side.sign()));
        skeleton_with_hand(upper, Some(forearm), palm, side)
    }

    /// Joint-space interpolation: slerp for the shoulder, linear for the rest.
    #[cfg(test)]
    pub fn lerp(a: &ArmState, b: &ArmState, t: f64) -> ArmState {
        let t = t.clamp(0.0, 1.0);
        ArmState {
            humerus: a.humerus.slerp(b.humerus, t),
            elbow_deg: a.elbow_deg + (b.elbow_deg - a.elbow_deg) * t,
            pronation_deg: a.pronation_deg + (b.pronation_deg - a.pronation_deg) * t,
        }
    }
}

/// Smooth value of a joint-angle trajectory at time `t`.
///
/// `keys` are `(time, value)` pairs with strictly increasing times (usually
/// 0 to 1). Between keys the value follows a monotone cubic (PCHIP), so it is
/// continuous, has a continuous slope, never overshoots a key and never
/// reverses direction except at a key where the trajectory itself turns.
pub fn track(keys: &[(f64, f64)], t: f64) -> f64 {
    let n = keys.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return keys[0].1;
    }
    let t = if t.is_finite() {
        t.clamp(keys[0].0, keys[n - 1].0)
    } else {
        keys[0].0
    };
    // Secant slopes and the tangent at each key.
    let secant: Vec<f64> = keys
        .windows(2)
        .map(|w| (w[1].1 - w[0].1) / (w[1].0 - w[0].0))
        .collect();
    let mut tangent = vec![0.0; n];
    tangent[0] = secant[0];
    tangent[n - 1] = secant[n - 2];
    for i in 1..n - 1 {
        if secant[i - 1] * secant[i] > 0.0 {
            let (h0, h1) = (keys[i].0 - keys[i - 1].0, keys[i + 1].0 - keys[i].0);
            let (w1, w2) = (2.0 * h1 + h0, h1 + 2.0 * h0);
            tangent[i] = (w1 + w2) / (w1 / secant[i - 1] + w2 / secant[i]);
        }
    }
    let i = keys.windows(2).position(|w| t <= w[1].0).unwrap_or(n - 2);
    let (t0, v0) = keys[i];
    let (t1, v1) = keys[i + 1];
    let h = t1 - t0;
    let x = (t - t0) / h;
    let (x2, x3) = (x * x, x * x * x);
    (2.0 * x3 - 3.0 * x2 + 1.0) * v0
        + (x3 - 2.0 * x2 + x) * h * tangent[i]
        + (-2.0 * x3 + 3.0 * x2) * v1
        + (x3 - x2) * h * tangent[i + 1]
}

/// The same keys played backwards in time (an extension from its flexion).
pub fn reversed_keys<const N: usize>(keys: [(f64, f64); N]) -> [(f64, f64); N] {
    let mut out = keys;
    for (i, (t, v)) in keys.iter().rev().enumerate() {
        out[i] = (1.0 - t, *v);
    }
    out
}

/// Elevation (deg from hanging) and plane of elevation (deg) of an upper-arm
/// direction, for reading back what a state means.
pub fn elevation_and_plane(upper_dir: Vec3, side: Side) -> (f64, f64) {
    let elevation = upper_dir.angle_to(Vec3::DOWN).to_degrees();
    let plane = upper_dir.y.atan2(side.sign() * upper_dir.x).to_degrees();
    (elevation, plane)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arm3d::HAND_LENGTH;
    use crate::kinematics::UPPER_SHARE;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-9
    }

    const SIDES: [Side; 2] = [Side::Right, Side::Left];

    fn pose(plane: f64, elevation: f64, axial: f64, elbow: f64, pronation: f64) -> JointPose {
        JointPose {
            plane_deg: plane,
            elevation_deg: elevation,
            axial_deg: axial,
            elbow_deg: elbow,
            pronation_deg: pronation,
        }
    }

    #[test]
    fn tracks_hit_their_keys_stay_continuous_and_never_overshoot() {
        let keys = [(0.0, 60.0), (0.5, 85.0), (1.0, 5.0)];
        for (t, v) in keys {
            assert!((track(&keys, t) - v).abs() < 1e-9, "{t}");
        }
        let mut prev = track(&keys, 0.0);
        let mut peak = prev;
        for i in 1..=1000 {
            let v = track(&keys, f64::from(i) / 1000.0);
            assert!((v - prev).abs() < 0.6, "continuous at {i}: {prev} -> {v}");
            assert!((5.0 - 1e-9..=85.0 + 1e-9).contains(&v), "no overshoot: {v}");
            peak = peak.max(v);
            prev = v;
        }
        assert!(
            (peak - 85.0).abs() < 1e-6,
            "turns at the middle key, not elsewhere"
        );
    }

    #[test]
    fn a_rising_track_never_dips_and_flat_ends_stay_flat() {
        let keys = [(0.0, 15.0), (0.5, 70.0), (1.0, 115.0)];
        let mut prev = track(&keys, 0.0);
        for i in 1..=500 {
            let v = track(&keys, f64::from(i) / 500.0);
            assert!(v >= prev - 1e-9, "monotone rise at {i}");
            prev = v;
        }
        assert_eq!(track(&[(0.0, 7.0)], 0.4), 7.0);
        assert_eq!(track(&[], 0.4), 0.0);
        assert_eq!(
            track(&[(0.0, 1.0), (1.0, 3.0)], 0.5),
            2.0,
            "two keys are linear"
        );
        assert_eq!(track(&keys, -3.0), 15.0);
        assert_eq!(track(&keys, 9.0), 115.0);
        assert_eq!(track(&keys, f64::NAN), 15.0);
    }

    #[test]
    fn reversed_keys_play_the_trajectory_backwards() {
        let keys = [(0.0, 15.0), (0.3, 50.0), (1.0, 115.0)];
        let back = reversed_keys(keys);
        for i in 0..=20 {
            let t = f64::from(i) / 20.0;
            assert!((track(&back, t) - track(&keys, 1.0 - t)).abs() < 1e-9);
        }
        assert!(
            back.windows(2).all(|w| w[0].0 < w[1].0),
            "times stay increasing"
        );
    }

    #[test]
    fn rest_pose_hangs_straight_down_at_full_length() {
        for side in SIDES {
            let sk = JointPose::REST.state(side).skeleton(side);
            assert!(close(sk.elbow, Vec3::new(0.0, 0.0, -UPPER_SHARE)));
            assert!(close(sk.wrist, Vec3::new(0.0, 0.0, -1.0)));
        }
    }

    #[test]
    fn segment_lengths_are_constant_for_every_pose() {
        for side in SIDES {
            for plane in [-60.0, 0.0, 45.0, 90.0, 150.0, 180.0] {
                for elevation in [0.0, 30.0, 90.0, 135.0, 180.0] {
                    for (axial, elbow, pron) in
                        [(0.0, 0.0, 0.0), (-40.0, 60.0, 30.0), (40.0, 145.0, -60.0)]
                    {
                        let sk = pose(plane, elevation, axial, elbow, pron)
                            .state(side)
                            .skeleton(side);
                        assert!(((sk.elbow - sk.shoulder).length() - UPPER_SHARE).abs() < 1e-9);
                        assert!(
                            ((sk.wrist - sk.elbow).length() - (1.0 - UPPER_SHARE)).abs() < 1e-9
                        );
                        assert!(
                            sk.wrist.length() <= 1.0 + 1e-9,
                            "reach never exceeds the arm"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn plane_and_elevation_place_the_upper_arm_where_they_say() {
        for side in SIDES {
            let s = side.sign();
            let up = |plane, elevation| {
                pose(plane, elevation, 0.0, 0.0, 0.0)
                    .state(side)
                    .directions()
                    .0
            };
            assert!(
                close(up(0.0, 90.0), Vec3::new(s, 0.0, 0.0)),
                "abduction goes out to the side"
            );
            assert!(close(up(90.0, 90.0), Vec3::Y), "flexion goes forward");
            assert!(
                close(up(180.0, 90.0), Vec3::new(-s, 0.0, 0.0)),
                "adduction crosses the body"
            );
            assert!(close(up(-90.0, 90.0), -Vec3::Y), "extension goes backward");
            assert!(
                close(up(33.0, 180.0), Vec3::Z),
                "overhead is overhead for any plane"
            );
            let (e, p) = elevation_and_plane(up(70.0, 50.0), side);
            assert!((e - 50.0).abs() < 1e-9 && (p - 70.0).abs() < 1e-9);
        }
    }

    #[test]
    fn isolated_elbow_flexion_is_a_hinge_about_the_lateral_axis() {
        for side in SIDES {
            for elbow in [0.0, 30.0, 90.0, 145.0] {
                let st = pose(0.0, 0.0, 0.0, elbow, 0.0).state(side);
                let (upper, forearm) = st.directions();
                assert!(close(upper, Vec3::DOWN), "the upper arm does not move");
                // Anatomical position: the forearm swings forward in the sagittal plane.
                let e = elbow.to_radians();
                assert!(
                    close(forearm, Vec3::new(0.0, e.sin(), -e.cos())),
                    "{side:?} {elbow}"
                );
                assert!(close(st.hinge_axis(), Vec3::X), "hinge axis is lateral");
                // Flexion angle between the segments equals the elbow angle.
                assert!((upper.angle_to(forearm).to_degrees() - elbow).abs() < 1e-9);
                // The forearm never leaves the plane perpendicular to the hinge axis.
                assert!(forearm.dot(st.hinge_axis()).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn the_elbow_never_bends_sideways_in_any_shoulder_pose() {
        // Hinge: the forearm always stays in the plane perpendicular to the hinge axis.
        for side in SIDES {
            for plane in [-45.0, 0.0, 60.0, 120.0, 180.0] {
                for elevation in [20.0, 90.0, 160.0] {
                    for axial in [-60.0, 0.0, 60.0] {
                        let st = pose(plane, elevation, axial, 80.0, 25.0).state(side);
                        let (upper, forearm) = st.directions();
                        assert!(forearm.dot(st.hinge_axis()).abs() < 1e-9);
                        assert!(upper.dot(st.hinge_axis()).abs() < 1e-9);
                        assert!((upper.angle_to(forearm).to_degrees() - 80.0).abs() < 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn axial_rotation_turns_the_bending_plane_but_not_a_straight_arm() {
        for side in SIDES {
            let straight = |axial| {
                pose(40.0, 100.0, axial, 0.0, 0.0)
                    .state(side)
                    .skeleton(side)
                    .wrist
            };
            assert!(
                close(straight(-50.0), straight(50.0)),
                "a straight arm's wrist does not move"
            );
            // With the elbow bent, external rotation moves the hand around the upper-arm axis.
            let bent = |axial| {
                pose(0.0, 90.0, axial, 90.0, 0.0)
                    .state(side)
                    .skeleton(side)
                    .wrist
            };
            assert!((bent(0.0) - bent(60.0)).length() > 0.3);
            // 90 deg abduction, elbow at 90: neutral rotation points the hand forward,
            // external rotation points it up ("cactus").
            let hand_dir = |axial| pose(0.0, 90.0, axial, 90.0, 0.0).state(side).directions().1;
            assert!(
                close(hand_dir(0.0), Vec3::Y),
                "{side:?}: {:?}",
                hand_dir(0.0)
            );
            assert!(
                close(hand_dir(90.0), Vec3::Z),
                "{side:?}: external rotation lifts the hand: {:?}",
                hand_dir(90.0)
            );
            assert!(
                close(hand_dir(-90.0), -Vec3::Z),
                "{side:?}: internal rotation lowers it"
            );
        }
    }

    #[test]
    fn forearm_rotation_turns_the_palm_and_leaves_the_wrist_alone() {
        for side in SIDES {
            for (plane, elevation, elbow) in
                [(90.0, 60.0, 0.0), (30.0, 100.0, 70.0), (150.0, 40.0, 110.0)]
            {
                let at = |pron| {
                    pose(plane, elevation, 20.0, elbow, pron)
                        .state(side)
                        .skeleton(side)
                };
                let neutral = at(0.0);
                for pron in [-80.0, -30.0, 45.0, 90.0] {
                    let turned = at(pron);
                    assert!(close(turned.elbow, neutral.elbow));
                    assert!(
                        close(turned.wrist, neutral.wrist),
                        "the wrist does not orbit"
                    );
                    assert!(close(turned.forearm_dir, neutral.forearm_dir));
                    assert!(turned.palm_normal.dot(neutral.palm_normal) < 0.999 || pron == 0.0);
                    let angle = turned
                        .palm_normal
                        .angle_to(neutral.palm_normal)
                        .to_degrees();
                    assert!(
                        (angle - pron.abs()).abs() < 1e-6,
                        "{side:?} {pron}: palm turned {angle}"
                    );
                }
            }
        }
    }

    #[test]
    fn humeral_rotation_turns_the_palm_with_the_forearm_and_leaves_the_wrist_alone() {
        for side in SIDES {
            let at = |axial| {
                pose(40.0, 100.0, axial, 0.0, 0.0)
                    .state(side)
                    .skeleton(side)
            };
            let (a, b) = (at(0.0), at(50.0));
            assert!(
                close(a.wrist, b.wrist),
                "a straight arm's wrist does not move"
            );
            let turned = a.palm_normal.angle_to(b.palm_normal).to_degrees();
            assert!(
                (turned - 50.0).abs() < 1e-6,
                "{side:?}: the palm turned {turned}"
            );
            // With the elbow bent the same rotation also swings the hand around the upper arm.
            let bent = |axial| {
                pose(40.0, 100.0, axial, 80.0, 0.0)
                    .state(side)
                    .skeleton(side)
            };
            assert!((bent(0.0).wrist - bent(50.0).wrist).length() > 0.2);
            // Forearm roll and humeral rotation are different joints: the same palm turn,
            // but only the roll changes the forearm angle itself.
            let roll = pose(40.0, 100.0, 0.0, 0.0, 50.0).state(side);
            assert!((roll.pronation_deg - 50.0).abs() < 1e-12);
        }
    }

    #[test]
    fn the_palm_follows_the_full_forearm_orientation() {
        for side in SIDES {
            for (plane, elevation, axial, elbow, pron) in [
                (150.0, 35.0, -40.0, 45.0, 20.0),
                (30.0, 135.0, 40.0, 5.0, -50.0),
                (125.0, 120.0, 40.0, 40.0, -45.0),
            ] {
                let st = pose(plane, elevation, axial, elbow, pron).state(side);
                let sk = st.skeleton(side);
                let q = st.forearm_orientation(side);
                assert!(close(sk.palm_normal, q.rotate(Vec3::X.scale(-side.sign()))));
                assert!(
                    close(sk.thumb_dir, q.rotate(Vec3::Y)),
                    "{side:?}: thumb {:?}",
                    sk.thumb_dir
                );
                assert!(close(sk.forearm_dir, q.rotate(Vec3::DOWN)));
            }
        }
    }

    #[test]
    fn pronation_turns_the_palm_down_with_the_forearm_forward() {
        for side in SIDES {
            let palm = |pron| {
                pose(90.0, 90.0, 0.0, 0.0, pron)
                    .state(side)
                    .skeleton(side)
                    .palm_normal
            };
            assert!(palm(90.0).z < -0.99, "pronated: palm down");
            assert!(palm(-90.0).z > 0.99, "supinated: palm up");
        }
    }

    #[test]
    fn left_arm_poses_are_mirror_images_of_the_right() {
        let mirror = |v: Vec3| Vec3::new(-v.x, v.y, v.z);
        for (plane, elevation, axial, elbow, pron) in [
            (150.0, 35.0, -40.0, 0.0, 20.0),
            (30.0, 135.0, 40.0, 25.0, -45.0),
            (90.0, 70.0, 10.0, 100.0, 60.0),
        ] {
            let p = pose(plane, elevation, axial, elbow, pron);
            let right = p.state(Side::Right).skeleton(Side::Right);
            let left = p.state(Side::Left).skeleton(Side::Left);
            assert!(close(mirror(right.elbow), left.elbow));
            assert!(close(mirror(right.wrist), left.wrist));
            assert!(close(mirror(right.forearm_dir), left.forearm_dir));
            assert!(close(mirror(right.palm_normal), left.palm_normal));
            assert!(
                close(mirror(right.thumb_dir), left.thumb_dir),
                "thumb side mirrors too"
            );
        }
    }

    #[test]
    fn interpolation_is_smooth_and_stays_on_the_arm() {
        for side in SIDES {
            let a = pose(150.0, 35.0, -40.0, 0.0, 20.0).state(side);
            let b = pose(30.0, 135.0, 40.0, 0.0, -45.0).state(side);
            let mut prev: Option<ArmSkeleton> = None;
            for i in 0..=60 {
                let t = f64::from(i) / 60.0;
                let sk = ArmState::lerp(&a, &b, t).skeleton(side);
                assert!(((sk.wrist - sk.elbow).length() - (1.0 - UPPER_SHARE)).abs() < 1e-9);
                assert!(sk.wrist.length() <= 1.0 + 1e-9);
                if let Some(p) = prev {
                    assert!((sk.wrist - p.wrist).length() < 0.07, "no jumps: step {i}");
                    assert!(
                        sk.palm_normal.angle_to(p.palm_normal).to_degrees() < 4.0,
                        "no sudden palm flips"
                    );
                    assert!(sk.upper_dir.angle_to(p.upper_dir).to_degrees() < 4.0);
                }
                prev = Some(sk);
            }
            assert!(close(
                ArmState::lerp(&a, &b, 0.0).skeleton(side).wrist,
                a.skeleton(side).wrist
            ));
            assert!(close(
                ArmState::lerp(&a, &b, 1.0).skeleton(side).wrist,
                b.skeleton(side).wrist
            ));
        }
    }

    #[test]
    fn limits_clamp_simulated_poses_and_repair_non_finite_values() {
        let limits = JointLimits::DEFAULT;
        let wild = pose(f64::NAN, 400.0, -200.0, -30.0, f64::INFINITY);
        let fixed = limits.clamp(wild);
        assert_eq!(
            fixed,
            JointPose {
                plane_deg: 0.0,
                elevation_deg: 180.0,
                axial_deg: -90.0,
                elbow_deg: 0.0,
                pronation_deg: 0.0
            }
        );
        assert_eq!(
            limits.clamp(pose(10.0, 20.0, 30.0, 40.0, 50.0)),
            pose(10.0, 20.0, 30.0, 40.0, 50.0)
        );
        let tight = JointLimits {
            elbow: (0.0, 60.0),
            ..limits
        };
        assert_eq!(tight.clamp(pose(0.0, 0.0, 0.0, 120.0, 0.0)).elbow_deg, 60.0);
    }

    #[test]
    fn the_hand_extends_the_forearm_from_the_wrist() {
        for side in SIDES {
            let sk = pose(60.0, 80.0, 10.0, 40.0, 30.0)
                .state(side)
                .skeleton(side);
            assert!(close(
                sk.hand_tip,
                sk.wrist + sk.forearm_dir.scale(HAND_LENGTH)
            ));
            assert!(sk.forearm_measured);
        }
    }
}
