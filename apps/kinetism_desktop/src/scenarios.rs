//! Scripted simulator scenarios for the 3D game: synchronized upper-arm and
//! forearm motion for elbow, shoulder, forearm-rotation and PNF patterns,
//! including incorrect movements and sensor problems.
//!
//! Scenarios are pure functions of time, so they are easy to test and the
//! simulator just samples them. Diagonal scenarios play the joint-space
//! reference motion of [`crate::pnf3d`]: shoulder (slerp), elbow and forearm
//! rotation as three separate trajectories on one shared time parameter, so
//! the simulated arm moves like an arm: no inverse kinematics. Everything they drive is simulated data: the app marks it as
//! practice and never saves it as a patient measurement.

use crate::armpose::{ArmState, JointPose};
use crate::pnf3d::{PnfConfig, PnfPattern};

/// Seconds for one repetition of a scenario.
pub const CYCLE_SECONDS: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    ElbowFlexExt,
    ShoulderAbduction,
    ShoulderFlexion,
    ForearmRotation,
    D1Flexion,
    D1Extension,
    D2Flexion,
    D2Extension,
    /// D2 flexion with the forearm held neutral (shoulder and elbow only).
    D2NoRotation,
    /// D2 flexion that stops about halfway.
    IncompleteD2,
    /// The D1 diagonal while D2 is asked for.
    WrongDiagonal,
    /// All packets stop for a moment mid-movement.
    SensorDropout,
    /// Only the forearm sensor stops reporting.
    ForearmDropout,
    /// Jitter and corrupt packets.
    NoisyLink,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkQuality {
    Good,
    /// No packets at all.
    Silent,
    /// Packets carry only the upper arm.
    ForearmMissing,
    /// Small orientation noise and occasional corrupt packets.
    Noisy,
}

/// The simulated arm at one instant, plus the state of the link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScenarioPose {
    pub state: ArmState,
    pub quality: LinkQuality,
}

impl Scenario {
    pub const ALL: [Scenario; 14] = [
        Scenario::D2Flexion,
        Scenario::D2NoRotation,
        Scenario::D2Extension,
        Scenario::D1Flexion,
        Scenario::D1Extension,
        Scenario::ForearmRotation,
        Scenario::ElbowFlexExt,
        Scenario::ShoulderAbduction,
        Scenario::ShoulderFlexion,
        Scenario::IncompleteD2,
        Scenario::WrongDiagonal,
        Scenario::SensorDropout,
        Scenario::ForearmDropout,
        Scenario::NoisyLink,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Scenario::ElbowFlexExt => "Elbow flexion / extension",
            Scenario::ShoulderAbduction => "Shoulder abduction",
            Scenario::ShoulderFlexion => "Shoulder flexion",
            Scenario::ForearmRotation => "Forearm rotation",
            Scenario::D1Flexion => "D1 flexion",
            Scenario::D1Extension => "D1 extension",
            Scenario::D2Flexion => "D2 flexion",
            Scenario::D2Extension => "D2 extension",
            Scenario::D2NoRotation => "D2 flexion, forearm held neutral",
            Scenario::IncompleteD2 => "Incomplete D2 (stops halfway)",
            Scenario::WrongDiagonal => "Wrong diagonal (D1 instead of D2)",
            Scenario::SensorDropout => "Sensor dropout",
            Scenario::ForearmDropout => "Forearm sensor dropout",
            Scenario::NoisyLink => "Noisy link",
        }
    }

    fn quality_at(self, phase: f64) -> LinkQuality {
        match self {
            Scenario::SensorDropout if (2.0..3.5).contains(&phase) => LinkQuality::Silent,
            Scenario::ForearmDropout if (2.0..3.5).contains(&phase) => LinkQuality::ForearmMissing,
            Scenario::NoisyLink => LinkQuality::Noisy,
            _ => LinkQuality::Good,
        }
    }

    /// The pattern the scenario is meant to be played against.
    pub fn target_pattern(self) -> PnfPattern {
        match self {
            Scenario::D1Flexion => PnfPattern::D1Flexion,
            Scenario::D1Extension => PnfPattern::D1Extension,
            Scenario::D2Extension => PnfPattern::D2Extension,
            _ => PnfPattern::D2Flexion,
        }
    }

    /// Pose `t` seconds into the scenario (it repeats every [`CYCLE_SECONDS`]).
    /// Diagonal scenarios walk the movement the therapist configured (reach,
    /// start shift, held elbow), so a simulated patient matches any setting.
    pub fn sample_for(self, config: &PnfConfig, t: f64) -> ScenarioPose {
        let side = config.side;
        let phase = if t.is_finite() {
            t.rem_euclid(CYCLE_SECONDS)
        } else {
            0.0
        };
        let s = cycle_progress(phase);
        let joints = |pose: JointPose| pose.state(side);
        let pattern = |pattern: PnfPattern, progress: f64, rotate: bool| {
            PnfConfig { pattern, ..*config }.reference_state(progress, rotate)
        };
        let state = match self {
            Scenario::ElbowFlexExt => joints(JointPose {
                elbow_deg: 120.0 * s,
                ..JointPose::REST
            }),
            Scenario::ShoulderAbduction => joints(JointPose {
                elevation_deg: 120.0 * s,
                ..JointPose::REST
            }),
            Scenario::ShoulderFlexion => joints(JointPose {
                plane_deg: 90.0,
                elevation_deg: 120.0 * s,
                ..JointPose::REST
            }),
            Scenario::ForearmRotation => {
                // Arm held forward: sweep supination then pronation.
                let sweep = ((phase / CYCLE_SECONDS) * std::f64::consts::TAU).sin();
                joints(JointPose {
                    plane_deg: 90.0,
                    elevation_deg: 60.0,
                    pronation_deg: 70.0 * sweep,
                    ..JointPose::REST
                })
            }
            Scenario::D1Flexion => pattern(PnfPattern::D1Flexion, s, true),
            Scenario::D1Extension => pattern(PnfPattern::D1Extension, s, true),
            Scenario::D2Extension => pattern(PnfPattern::D2Extension, s, true),
            Scenario::D2NoRotation => pattern(PnfPattern::D2Flexion, s, false),
            Scenario::IncompleteD2 => pattern(PnfPattern::D2Flexion, s * 0.55, true),
            Scenario::WrongDiagonal => pattern(PnfPattern::D1Flexion, s, true),
            Scenario::D2Flexion
            | Scenario::SensorDropout
            | Scenario::ForearmDropout
            | Scenario::NoisyLink => pattern(PnfPattern::D2Flexion, s, true),
        };
        ScenarioPose {
            state,
            quality: self.quality_at(phase),
        }
    }
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// 0 at the start position, 1 at the end: hold, move out, hold, move back, hold.
pub fn cycle_progress(phase: f64) -> f64 {
    match phase {
        p if p < 1.0 => 0.0,
        p if p < 4.0 => smoothstep((p - 1.0) / 3.0),
        p if p < 4.5 => 1.0,
        p if p < 7.5 => 1.0 - smoothstep((p - 4.5) / 3.0),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arm3d::skeleton;
    use crate::clinical::Side;
    use crate::forearm::{RotationCalibration, RotationDirection, relative_orientation};
    use crate::kinematics::{
        Calibration, PoseAccumulator, SAMPLES_PER_POSE, SimArm, capture_forward, capture_rest,
    };
    use crate::math::Vec3;
    use crate::pnf3d::{
        Frame, Phase, PnfEvent, RotationOutcome, RotationRule, Round3d, Trajectory,
    };

    const SIDES: [Side; 2] = [Side::Right, Side::Left];

    fn calibrate(side: Side) -> (Calibration, RotationCalibration) {
        let mut sim = SimArm::new(side);
        let mut acc = PoseAccumulator::default();
        for _ in 0..SAMPLES_PER_POSE {
            let (u, f) = sim.quats();
            acc.push(u, Some(f));
        }
        let rest = capture_rest(&acc).unwrap();
        acc.clear();
        sim.elevation_deg = 60.0;
        sim.plane_deg = 90.0;
        for _ in 0..SAMPLES_PER_POSE {
            let (u, f) = sim.quats();
            acc.push(u, Some(f));
        }
        let cal = capture_forward(&rest, &acc).unwrap();
        let (u, f) = sim.quats();
        let rel = relative_orientation(u, f, cal.forearm_yaw_offset).unwrap();
        let rot = RotationCalibration::new(rel, cal.forearm_axis.unwrap(), cal.forearm_yaw_offset)
            .unwrap()
            .with_palm(cal.neutral_palm_in_sensor(f, side));
        (cal, rot)
    }

    /// The whole chain a live sensor goes through: quaternions, calibration,
    /// forward kinematics, rotation estimate, game frame.
    fn pipeline_frame(
        pose: &ScenarioPose,
        side: Side,
        cal: &Calibration,
        rot: &RotationCalibration,
    ) -> Option<Frame> {
        let mut arm = SimArm::new(side);
        arm.apply_state(&pose.state);
        let (u, f) = arm.quats();
        let forearm = (pose.quality != LinkQuality::ForearmMissing).then_some(f);
        if pose.quality == LinkQuality::Silent {
            return None;
        }
        let measured = cal.pose(u, forearm, side);
        let rotation = rot.estimate(u, forearm, side).ok().map(|r| r.angle_deg);
        let sk = hand_skeleton(&measured, rot, rotation, side);
        Frame::from_skeleton(&sk, rotation)
    }

    /// The arm the tracker draws: palm from the forearm sensor's full orientation.
    fn hand_skeleton(
        measured: &crate::kinematics::ArmPose,
        rot: &RotationCalibration,
        rotation: Option<f64>,
        side: Side,
    ) -> crate::arm3d::ArmSkeleton {
        match (measured.forearm_orientation, rot.palm_sensor()) {
            (Some(q), Some(palm)) => crate::arm3d::skeleton_with_hand(
                measured.upper_dir,
                measured.forearm_dir,
                q.rotate(palm),
                side,
            ),
            _ => skeleton(measured.upper_dir, measured.forearm_dir, rotation, side),
        }
    }

    fn truth(pose: &ScenarioPose, side: Side) -> Vec3 {
        pose.state.skeleton(side).wrist
    }

    #[test]
    fn the_estimated_wrist_follows_the_forward_kinematics_model() {
        // Sensor quaternions -> calibration -> FK reproduces the true wrist.
        for side in SIDES {
            let (cal, rot) = calibrate(side);
            let config = PnfConfig::new(PnfPattern::D2Flexion, side);
            for scenario in Scenario::ALL {
                if matches!(scenario, Scenario::SensorDropout | Scenario::ForearmDropout) {
                    continue;
                }
                for i in 0..16 {
                    let t = 0.5 * f64::from(i);
                    let pose = scenario.sample_for(&config, t);
                    let frame = pipeline_frame(&pose, side, &cal, &rot).unwrap();
                    assert!(
                        (frame.wrist - truth(&pose, side)).length() < 0.02,
                        "{side:?} {scenario:?} t={t}: {:?} vs {:?}",
                        frame.wrist,
                        truth(&pose, side)
                    );
                }
            }
        }
    }

    #[test]
    fn the_simulator_and_the_game_use_the_same_positions() {
        for pattern in PnfPattern::ALL {
            for side in SIDES {
                let config = PnfConfig {
                    reach: 0.8,
                    start_shift_deg: [5.0, 10.0],
                    ..PnfConfig::new(pattern, side)
                };
                let path = Trajectory::from_config(&config);
                let scenario = match pattern {
                    PnfPattern::D1Flexion => Scenario::D1Flexion,
                    PnfPattern::D1Extension => Scenario::D1Extension,
                    PnfPattern::D2Flexion => Scenario::D2Flexion,
                    PnfPattern::D2Extension => Scenario::D2Extension,
                };
                assert_eq!(scenario.target_pattern(), pattern);
                for i in 0..=12 {
                    // Progress along the cycle -> progress along the path (hold, move, ...).
                    let t = 1.0 + 3.0 * f64::from(i) / 12.0;
                    let progress = cycle_progress(t);
                    let wrist = truth(&scenario.sample_for(&config, t), side);
                    assert!(
                        (wrist - path.point(progress)).length() < 0.02,
                        "{pattern:?} {side:?} t={t}: {wrist:?} vs {:?}",
                        path.point(progress)
                    );
                }
            }
        }
    }

    const PATTERN_SCENARIOS: [Scenario; 5] = [
        Scenario::D1Flexion,
        Scenario::D1Extension,
        Scenario::D2Flexion,
        Scenario::D2Extension,
        Scenario::D2NoRotation,
    ];

    #[test]
    fn every_simulated_joint_angle_stays_continuous() {
        let dt = 0.05;
        for side in SIDES {
            let config = PnfConfig::new(PnfPattern::D2Flexion, side);
            for scenario in Scenario::ALL {
                let mut prev: Option<ScenarioPose> = None;
                for i in 0..400 {
                    let pose = scenario.sample_for(&config, dt * f64::from(i));
                    if let Some(p) = prev {
                        let who = format!("{scenario:?} {side:?} frame {i}");
                        assert!(
                            (pose.state.elbow_deg - p.state.elbow_deg).abs() < 6.0,
                            "elbow step {who}"
                        );
                        assert!(
                            (pose.state.pronation_deg - p.state.pronation_deg).abs() < 6.0,
                            "forearm step {who}"
                        );
                        let turn = (p.state.humerus.conjugate() * pose.state.humerus)
                            .w
                            .abs()
                            .clamp(0.0, 1.0)
                            .acos()
                            * 2.0;
                        assert!(turn.to_degrees() < 6.0, "shoulder step {who}");
                        let (a, b) = (p.state.skeleton(side), pose.state.skeleton(side));
                        assert!((a.wrist - b.wrist).length() < 0.09, "wrist jump {who}");
                    }
                    prev = Some(pose);
                }
            }
        }
    }

    #[test]
    fn pattern_scenarios_bend_and_straighten_the_elbow_with_the_shoulder() {
        let config = PnfConfig::new(PnfPattern::D2Flexion, Side::Right);
        let elbow_range = |scenario: Scenario| {
            let v: Vec<f64> = (0..160)
                .map(|i| {
                    scenario
                        .sample_for(&config, 0.05 * f64::from(i))
                        .state
                        .elbow_deg
                })
                .collect();
            v.iter().fold(f64::MIN, |a, b| a.max(*b)) - v.iter().fold(f64::MAX, |a, b| a.min(*b))
        };
        for scenario in PATTERN_SCENARIOS {
            assert!(
                elbow_range(scenario) > 25.0,
                "{scenario:?}: the elbow visibly bends and straightens"
            );
        }
        // Straight-elbow variation is a separate setting, not forced on or off.
        let straight = PnfConfig {
            elbow_mode: crate::pnf3d::ElbowMode::Straight,
            ..config
        };
        let flat: Vec<f64> = (0..160)
            .map(|i| {
                Scenario::D2Flexion
                    .sample_for(&straight, 0.05 * f64::from(i))
                    .state
                    .elbow_deg
            })
            .collect();
        assert!(flat.iter().all(|e| e.abs() < 1e-12));
        // At the start of a D2 swipe the elbow is bent; at the end it is nearly straight.
        let at = |t: f64| Scenario::D2Flexion.sample_for(&config, t).state.elbow_deg;
        assert!(at(0.5) > 30.0, "bent at the start: {}", at(0.5));
        assert!(
            at(4.2) < 30.0 && at(4.2) < at(0.5),
            "the elbow has straightened by the end of the reach: {} (start {})",
            at(4.2),
            at(0.5)
        );
    }

    #[test]
    fn the_palm_and_elbow_seen_by_the_sensors_match_the_simulated_joints() {
        // quaternions -> calibration -> estimates must agree with what was simulated,
        // through the whole diagonal, with a bent elbow and humeral rotation in play.
        for side in SIDES {
            let (cal, rot) = calibrate(side);
            let config = PnfConfig::new(PnfPattern::D2Flexion, side);
            for scenario in PATTERN_SCENARIOS {
                for i in 0..48 {
                    let t = 0.2 * f64::from(i);
                    let pose = scenario.sample_for(&config, t);
                    let mut arm = SimArm::new(side);
                    arm.apply_state(&pose.state);
                    let (u, f) = arm.quats();
                    let who = format!("{scenario:?} {side:?} t={t:.1}");
                    let measured = cal.pose(u, Some(f), side);
                    assert!(
                        (measured.elbow_flexion_deg.unwrap() - pose.state.elbow_deg).abs() < 1.0,
                        "elbow {who}: {:?} vs {}",
                        measured.elbow_flexion_deg,
                        pose.state.elbow_deg
                    );
                    let est = rot.estimate(u, Some(f), side).expect("a reliable estimate");
                    assert!(
                        (est.angle_deg - pose.state.pronation_deg).abs() < 1.0,
                        "forearm rotation {who}: {} vs {}",
                        est.angle_deg,
                        pose.state.pronation_deg
                    );
                    // The palm the renderer gets is the simulated palm.
                    let shown = hand_skeleton(&measured, &rot, Some(est.angle_deg), side);
                    let truth = pose.state.skeleton(side);
                    assert!(
                        shown.palm_normal.angle_to(truth.palm_normal).to_degrees() < 1.5,
                        "palm {who}"
                    );
                    assert!(
                        shown.thumb_dir.angle_to(truth.thumb_dir).to_degrees() < 1.5,
                        "thumb {who}"
                    );
                    assert!((shown.wrist - truth.wrist).length() < 0.02, "wrist {who}");
                }
            }
        }
    }

    #[test]
    fn the_palm_visibly_turns_during_the_diagonals_and_stays_put_without_forearm_rotation() {
        let config = PnfConfig::new(PnfPattern::D2Flexion, Side::Right);
        let palm_turn = |scenario: Scenario| {
            let normals: Vec<Vec3> = (0..160)
                .map(|i| {
                    scenario
                        .sample_for(&config, 0.05 * f64::from(i))
                        .state
                        .skeleton(Side::Right)
                        .palm_normal
                })
                .collect();
            let mut most = 0.0_f64;
            for a in &normals {
                for b in &normals {
                    most = most.max(a.angle_to(*b).to_degrees());
                }
            }
            most
        };
        assert!(
            palm_turn(Scenario::D2Flexion) > 60.0,
            "the palm rotates with the forearm"
        );
        assert!(palm_turn(Scenario::D1Flexion) > 60.0);
        // The palm also turns with the shoulder, but the forearm angle itself stays 0.
        let neutral: Vec<f64> = (0..160)
            .map(|i| {
                Scenario::D2NoRotation
                    .sample_for(&config, 0.05 * f64::from(i))
                    .state
                    .pronation_deg
            })
            .collect();
        assert!(neutral.iter().all(|p| *p == 0.0));
    }

    #[test]
    fn left_arm_scenarios_mirror_the_right_arm() {
        for scenario in Scenario::ALL {
            let right = PnfConfig::new(scenario.target_pattern(), Side::Right);
            let left = PnfConfig::new(scenario.target_pattern(), Side::Left);
            for i in 0..16 {
                let t = 0.5 * f64::from(i);
                let r = truth(&scenario.sample_for(&right, t), Side::Right);
                let l = truth(&scenario.sample_for(&left, t), Side::Left);
                assert!(
                    (Vec3::new(-r.x, r.y, r.z) - l).length() < 1e-9,
                    "{scenario:?} t={t}"
                );
            }
        }
    }

    #[test]
    fn upper_arm_and_forearm_rotations_are_independently_visible() {
        let side = Side::Right;
        let (cal, rot) = calibrate(side);
        let at = |plane, elevation, axial, pron| ScenarioPose {
            state: JointPose {
                plane_deg: plane,
                elevation_deg: elevation,
                axial_deg: axial,
                elbow_deg: 0.0,
                pronation_deg: pron,
            }
            .state(side),
            quality: LinkQuality::Good,
        };
        let frame = |p: &ScenarioPose| pipeline_frame(p, side, &cal, &rot).unwrap();
        let neutral = frame(&at(90.0, 60.0, 0.0, 0.0));
        let turned = frame(&at(90.0, 60.0, 0.0, 70.0));
        assert!(
            (neutral.wrist - turned.wrist).length() < 1e-6,
            "forearm roll leaves the wrist where it is"
        );
        assert!((turned.rotation_deg.unwrap() - 70.0).abs() < 0.5);
        let shoulder = frame(&at(90.0, 100.0, 0.0, 0.0));
        assert!((shoulder.wrist - neutral.wrist).length() > 0.3);
        assert!(
            shoulder.rotation_deg.unwrap().abs() < 0.5,
            "shoulder motion reads no forearm rotation"
        );
        // Humeral rotation turns upper arm and forearm together: no forearm-rotation reading.
        let axial = frame(&at(90.0, 60.0, 50.0, 0.0));
        assert!(
            axial.rotation_deg.unwrap().abs() < 0.5,
            "{:?}",
            axial.rotation_deg
        );
        assert!(
            (axial.wrist - neutral.wrist).length() < 1e-6,
            "a straight arm's wrist does not move"
        );
    }

    fn play(scenario: Scenario, rule: Option<RotationRule>, cycles: u32) -> Round3d {
        let side = Side::Right;
        let (cal, rot) = calibrate(side);
        let config = PnfConfig {
            reps: 3,
            rotation: rule,
            ..PnfConfig::new(scenario.target_pattern(), side)
        };
        let mut round = Round3d::new(config);
        let dt = 0.05;
        let mut t = 0.0;
        while t < CYCLE_SECONDS * f64::from(cycles) {
            let pose = scenario.sample_for(&config, t);
            round.update(pipeline_frame(&pose, side, &cal, &rot), dt);
            t += dt;
        }
        round
    }

    const SUPINATION_RULE: RotationRule = RotationRule {
        expected: RotationDirection::Supination,
        min_deg: 30.0,
        max_deg: 100.0,
    };

    #[test]
    fn every_diagonal_scenario_slices_fruit_along_the_3d_path() {
        for scenario in [
            Scenario::D1Flexion,
            Scenario::D1Extension,
            Scenario::D2Flexion,
            Scenario::D2Extension,
        ] {
            let round = play(scenario, None, 3);
            assert_eq!(round.scored_reps(), 3, "{scenario:?}: {:?}", round.results);
            assert_eq!(round.phase, Phase::Done);
            assert!(
                round
                    .results
                    .iter()
                    .all(|r| r.reached_top && r.fruits_sliced == r.fruits_total),
                "{scenario:?}"
            );
            assert!(round.spatial_fraction() > 0.9, "{scenario:?}");
        }
    }

    #[test]
    fn the_game_works_without_rotation_and_rotation_adds_a_separate_result() {
        let plain = play(Scenario::D2Flexion, None, 3);
        assert_eq!(plain.scored_reps(), 3);
        assert!(
            plain
                .results
                .iter()
                .all(|r| r.rotation == RotationOutcome::Disabled)
        );
        assert_eq!(plain.rotation_rate(), None);

        let rotated = play(Scenario::D2Flexion, Some(SUPINATION_RULE), 3);
        assert_eq!(rotated.scored_reps(), 3);
        assert_eq!(rotated.rotation_rate(), Some(1.0));
        assert!((rotated.spatial_fraction() - plain.spatial_fraction()).abs() < 0.02);

        // Same movement with the forearm held neutral: spatial success, rotation not reached.
        let missing = play(Scenario::D2NoRotation, Some(SUPINATION_RULE), 3);
        assert_eq!(missing.scored_reps(), 3);
        assert!(
            missing
                .results
                .iter()
                .all(|r| r.reached_top && r.rotation == RotationOutcome::NotReached)
        );
        assert_eq!(missing.rotation_rate(), Some(0.0));
        assert!((missing.spatial_fraction() - plain.spatial_fraction()).abs() < 0.02);
    }

    #[test]
    fn incorrect_or_incomplete_movements_do_not_score_as_successes() {
        let incomplete = play(Scenario::IncompleteD2, None, 3);
        assert!(incomplete.results.iter().all(|r| !r.reached_top));
        assert_eq!(incomplete.scored_reps(), 0, "{:?}", incomplete.results);

        let wrong = play(Scenario::WrongDiagonal, None, 3);
        assert!(
            wrong
                .results
                .iter()
                .all(|r| !r.reached_top || r.fruits_sliced < r.fruits_total),
            "{:?}",
            wrong.results
        );
        assert_eq!(wrong.scored_reps(), 0);
    }

    #[test]
    fn sensor_problems_never_produce_a_scored_rep() {
        for scenario in [Scenario::SensorDropout, Scenario::ForearmDropout] {
            for rule in [None, Some(SUPINATION_RULE)] {
                let round = play(scenario, rule, 2);
                assert_eq!(
                    round.scored_reps(),
                    0,
                    "{scenario:?} {rule:?}: {:?}",
                    round.results
                );
                assert!(round.discarded >= 1, "{scenario:?} {rule:?}");
            }
        }
    }

    #[test]
    fn an_invalid_calibration_or_estimate_cannot_score() {
        // Orientations the calibration cannot use give no frame, so nothing is scored.
        let mut round = Round3d::new(PnfConfig::new(PnfPattern::D2Flexion, Side::Right));
        for _ in 0..200 {
            assert!(
                round
                    .update(None, 0.05)
                    .iter()
                    .all(|e| matches!(e, PnfEvent::TrackingLost))
            );
        }
        assert_eq!((round.scored_reps(), round.rep, round.points), (0, 0, 0));
        // A missing forearm is lost tracking, not a guessed wrist.
        let sk = skeleton(Vec3::X, None, None, Side::Right);
        assert_eq!(Frame::from_skeleton(&sk, None), None);
        let sk = skeleton(Vec3::X, Some(Vec3::X), None, Side::Right);
        assert!(Frame::from_skeleton(&sk, None).is_some());
    }

    #[test]
    fn collision_uses_true_3d_positions_not_a_projection() {
        // A fruit directly behind the blade in depth (same x/z, different y) is not sliced.
        let side = Side::Right;
        let mut round = Round3d::new(PnfConfig::new(PnfPattern::D2Flexion, side));
        let path = round.trajectory;
        let fruit = round.fruits[0].pos;
        let frame = |wrist: Vec3, blade_tip: Vec3| Frame {
            wrist,
            blade_tip,
            rotation_deg: None,
        };
        for _ in 0..8 {
            round.update(
                Some(frame(
                    path.start,
                    path.start + path.tangent(0.0).scale(0.42),
                )),
                0.05,
            );
        }
        let behind = Vec3::new(fruit.x, fruit.y - 0.5, fruit.z);
        let ev = round.update(Some(frame(behind, behind + Vec3::new(0.0, 0.2, 0.0))), 0.05);
        assert!(
            !ev.iter().any(|e| matches!(e, PnfEvent::Sliced { .. })),
            "a fruit in front of or behind the blade is not hit"
        );
        let on_it = round.update(Some(frame(path.point(0.35), fruit)), 0.05);
        assert!(
            on_it.iter().any(|e| matches!(e, PnfEvent::Sliced { .. })),
            "{on_it:?}"
        );
    }

    #[test]
    fn every_scenario_produces_finite_valid_arm_states_for_both_arms() {
        for scenario in Scenario::ALL {
            for side in SIDES {
                let config = PnfConfig::new(scenario.target_pattern(), side);
                for i in 0..40 {
                    let p = scenario.sample_for(&config, 0.25 * f64::from(i));
                    assert!(p.state.elbow_deg.is_finite() && p.state.pronation_deg.is_finite());
                    assert!((p.state.humerus.norm() - 1.0).abs() < 1e-9);
                    assert!(
                        (0.0..=145.0).contains(&p.state.elbow_deg),
                        "{scenario:?}: elbow limits"
                    );
                }
            }
        }
        let config = PnfConfig::new(PnfPattern::D2Flexion, Side::Right);
        assert!(
            Scenario::D2Flexion
                .sample_for(&config, f64::NAN)
                .state
                .humerus
                .norm()
                .is_finite()
        );
    }

    #[test]
    fn scenarios_cover_the_requested_motions() {
        let config = PnfConfig::new(PnfPattern::D2Flexion, Side::Right);
        let max = |s: Scenario, f: fn(&ScenarioPose) -> f64| {
            (0..64)
                .map(|i| f(&s.sample_for(&config, 0.125 * f64::from(i))).abs())
                .fold(0.0, f64::max)
        };
        assert!(max(Scenario::ElbowFlexExt, |p| p.state.elbow_deg) > 100.0);
        assert!(max(Scenario::ForearmRotation, |p| p.state.pronation_deg) > 60.0);
        assert!(max(Scenario::D2Flexion, |p| p.state.pronation_deg) > 40.0);
        assert!(
            max(Scenario::ShoulderAbduction, |p| {
                p.state.directions().0.angle_to(Vec3::DOWN).to_degrees()
            }) > 100.0
        );
        // Shoulder flexion goes forward, abduction goes out to the side.
        let flexion = Scenario::ShoulderFlexion
            .sample_for(&config, 4.2)
            .state
            .directions()
            .0;
        assert!(flexion.y > 0.5 && flexion.x.abs() < 0.1);
        let abduction = Scenario::ShoulderAbduction
            .sample_for(&config, 4.2)
            .state
            .directions()
            .0;
        assert!(abduction.x > 0.5 && abduction.y.abs() < 0.1);
        let qualities: Vec<_> = (0..64)
            .map(|i| {
                Scenario::SensorDropout
                    .sample_for(&config, 0.125 * f64::from(i))
                    .quality
            })
            .collect();
        assert!(qualities.contains(&LinkQuality::Silent) && qualities.contains(&LinkQuality::Good));
        assert_eq!(
            Scenario::NoisyLink.sample_for(&config, 1.0).quality,
            LinkQuality::Noisy
        );
    }

    #[test]
    fn scored_reps_are_reported_through_events_with_both_components() {
        let side = Side::Right;
        let (cal, rot) = calibrate(side);
        let config = PnfConfig {
            reps: 1,
            rotation: Some(SUPINATION_RULE),
            ..PnfConfig::new(PnfPattern::D2Flexion, side)
        };
        let mut round = Round3d::new(config);
        let mut scored = None;
        let mut t = 0.0;
        while t < CYCLE_SECONDS {
            let pose = Scenario::D2Flexion.sample_for(&config, t);
            for e in round.update(pipeline_frame(&pose, side, &cal, &rot), 0.05) {
                if let PnfEvent::RepScored(r) = e {
                    scored = Some(r);
                }
            }
            t += 0.05;
        }
        let rep = scored.expect("a scored rep");
        assert!(rep.reached_top && rep.rotation == RotationOutcome::Reached);
        assert!(rep.peak_rotation_deg.unwrap() > 30.0);
    }
}
