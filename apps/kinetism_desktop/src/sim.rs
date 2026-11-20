//! Simulated `Physio_A` sensor for demos without hardware.
//!
//! Generates real quaternion packets from a virtual arm and feeds them through
//! the same parser and calibration pipeline as BLE data.

use crate::clinical::Side;
use crate::game::Pt;
use crate::kinematics::SimArm;
use crate::math::{Quat, Vec3};
use crate::pnf3d::PnfConfig;
use crate::scenarios::{LinkQuality, Scenario};
use crate::sensor::SensorHub;
use crate::sensor::frame::encode_frame;

const PACKET_INTERVAL: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimInput {
    /// Elevation / plane / elbow sliders.
    Sliders,
    /// Hand follows a target point (mouse pointer or autopilot).
    HandTarget(Pt),
    /// A scripted 3D scenario (see [`SimDriver::start_scenario`]).
    Scenario,
}

/// Orientation noise (rad) added to every packet of a noisy link.
const NOISE_RAD: f64 = 0.012;
/// Share of packets replaced by garbage on a noisy link.
const GARBAGE_CHANCE: f64 = 0.04;

pub struct SimDriver {
    pub arm: SimArm,
    pub input: SimInput,
    accum: f64,
    scenario: Option<(Scenario, PnfConfig, f64)>,
    /// The joint state the running scenario last simulated (ground truth, not a measurement).
    truth: Option<crate::armpose::ArmState>,
    quality: LinkQuality,
    rng: fastrand::Rng,
}

impl SimDriver {
    pub fn new(side: Side) -> Self {
        Self {
            arm: SimArm::new(side),
            input: SimInput::Sliders,
            accum: 0.0,
            scenario: None,
            truth: None,
            quality: LinkQuality::Good,
            rng: fastrand::Rng::new(),
        }
    }

    /// Play a scripted scenario from its start. `config` fixes the diagonal
    /// the simulated patient walks (the therapist's reach and start offset).
    pub fn start_scenario(&mut self, scenario: Scenario, config: PnfConfig) {
        self.scenario = Some((scenario, config, 0.0));
        self.input = SimInput::Scenario;
    }

    /// What the running scenario simulated last. For debugging and tests;
    /// the app itself only ever sees the sensor packets.
    pub fn truth(&self) -> Option<crate::armpose::ArmState> {
        self.truth
    }

    pub fn stop_scenario(&mut self) {
        self.scenario = None;
        self.truth = None;
        self.arm.clear_scripted_shoulder();
        self.quality = LinkQuality::Good;
        if self.input == SimInput::Scenario {
            self.input = SimInput::Sliders;
        }
    }

    fn noisy(&mut self, q: Quat) -> Quat {
        let axis = Vec3::new(
            self.rng.f64() - 0.5,
            self.rng.f64() - 0.5,
            self.rng.f64() - 0.5,
        );
        let angle = (self.rng.f64() - 0.5) * 2.0 * NOISE_RAD;
        (Quat::from_axis_angle(axis, angle) * q)
            .normalized()
            .unwrap_or(q)
    }

    pub fn set_side(&mut self, side: Side) {
        if self.arm.side != side {
            let mut arm = SimArm::new(side);
            arm.elevation_deg = self.arm.elevation_deg;
            arm.plane_deg = self.arm.plane_deg;
            arm.elbow_deg = self.arm.elbow_deg;
            arm.pronation_deg = self.arm.pronation_deg;
            arm.axial_deg = self.arm.axial_deg;
            self.arm = arm;
        }
    }

    pub fn rest_pose(&mut self) {
        self.input = SimInput::Sliders;
        self.arm.elevation_deg = 0.0;
        self.arm.plane_deg = 0.0;
        self.arm.elbow_deg = 0.0;
        self.arm.pronation_deg = 0.0;
        self.arm.axial_deg = 0.0;
        self.arm.clear_scripted_shoulder();
        self.stop_scenario();
    }

    pub fn forward_pose(&mut self) {
        self.input = SimInput::Sliders;
        self.arm.elevation_deg = 60.0;
        self.arm.plane_deg = 90.0;
        self.arm.elbow_deg = 0.0;
        self.arm.pronation_deg = 0.0;
        self.arm.axial_deg = 0.0;
        self.arm.clear_scripted_shoulder();
        self.stop_scenario();
    }

    /// Emit packets at the firmware rate (20 Hz).
    pub fn tick(&mut self, hub: &SensorHub, dt: f64) {
        if let SimInput::HandTarget(p) = self.input {
            self.arm.set_hand_target(p[0], p[1]);
        }
        if self.input == SimInput::Scenario
            && let Some((scenario, config, t)) = &mut self.scenario
        {
            *t += dt;
            let pose = scenario.sample_for(config, *t);
            self.arm.apply_state(&pose.state);
            self.truth = Some(pose.state);
            self.quality = pose.quality;
        }
        self.accum += dt;
        while self.accum >= PACKET_INTERVAL {
            self.accum -= PACKET_INTERVAL;
            let (u, f) = self.arm.quats();
            match self.quality {
                LinkQuality::Good => hub.push_simulated(&encode_frame(u, Some(f))),
                LinkQuality::Silent => {}
                LinkQuality::ForearmMissing => hub.push_simulated(&encode_frame(u, None)),
                LinkQuality::Noisy => {
                    if self.rng.f64() < GARBAGE_CHANCE {
                        hub.push_simulated("1,2,3");
                    } else {
                        let (u, f) = (self.noisy(u), self.noisy(f));
                        hub.push_simulated(&encode_frame(u, Some(f)));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::{LinkState, Source};

    #[test]
    fn simulator_streams_through_the_hub_until_the_source_changes() {
        // Never driven: the Bluetooth task is queued but cannot run, so no radio is used.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let hub = SensorHub::new(runtime.handle(), eframe::egui::Context::default(), 0);
        let mut sim = SimDriver::new(Side::Right);
        sim.tick(&hub, 0.2);
        assert!(
            hub.drain().is_empty(),
            "packets are dropped until the simulator is the source"
        );

        hub.start_simulation();
        assert_eq!(
            (hub.source(), hub.state()),
            (Source::Simulator, LinkState::Simulated)
        );
        sim.tick(&hub, 0.12);
        let frames = hub.drain();
        assert_eq!(frames.len(), 2, "20 Hz: two packets in 120 ms");
        assert!(
            frames
                .iter()
                .all(|f| f.frame.upper.is_some() && f.frame.forearm.is_some()),
            "both segments are simulated"
        );
        sim.tick(&hub, 0.01);
        assert!(hub.drain().is_empty(), "partial intervals carry over");
        sim.tick(&hub, 0.04);
        assert_eq!(hub.drain().len(), 1);
        assert_eq!(hub.stats().packets, 3);

        sim.input = SimInput::HandTarget([3.0, 4.0]);
        sim.tick(&hub, 0.01);
        let far = sim.arm;
        sim.arm.set_hand_target(0.6, 0.8);
        assert!(
            (far.elevation_deg - sim.arm.elevation_deg).abs() < 1e-9,
            "targets out of reach clamp to the arm's length"
        );
        assert!(
            (140.0..=145.0).contains(&far.elevation_deg),
            "hand up and out: {}",
            far.elevation_deg
        );

        sim.forward_pose();
        assert_eq!(
            sim.input,
            SimInput::Sliders,
            "presets take over from the mouse"
        );
        sim.arm.elbow_deg = 35.0;
        sim.set_side(Side::Left);
        assert_eq!(sim.arm.side, Side::Left);
        assert_eq!(
            (sim.arm.elevation_deg, sim.arm.plane_deg, sim.arm.elbow_deg),
            (60.0, 90.0, 35.0),
            "switching arms keeps the slider pose"
        );
        sim.rest_pose();
        assert_eq!(
            (sim.arm.elevation_deg, sim.arm.plane_deg, sim.arm.elbow_deg),
            (0.0, 0.0, 0.0)
        );

        hub.push_simulated("1,2,3");
        let stats = hub.stats();
        assert_eq!(stats.invalid_packets, 1);
        assert_eq!(
            stats.last_invalid.as_deref(),
            Some("expected 8 values, got 3")
        );

        hub.disconnect();
        assert_eq!((hub.source(), hub.state()), (Source::None, LinkState::Idle));
        assert_eq!(
            hub.stats().packets,
            0,
            "stats restart when the source changes"
        );
        sim.tick(&hub, 0.2);
        assert!(
            hub.drain().is_empty(),
            "late simulator packets are ignored after disconnecting"
        );
        hub.start_simulation();
        sim.tick(&hub, 0.06);
        assert_eq!(hub.drain().len(), 1, "simulation resumes cleanly");
    }

    #[test]
    fn scenarios_drive_both_segments_and_degrade_the_link_on_purpose() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let hub = SensorHub::new(runtime.handle(), eframe::egui::Context::default(), 0);
        hub.start_simulation();
        let config = PnfConfig::new(crate::pnf3d::PnfPattern::D2Flexion, Side::Right);
        let mut sim = SimDriver::new(Side::Right);

        sim.start_scenario(Scenario::D2Flexion, config);
        assert_eq!(sim.input, SimInput::Scenario);
        // Hold at the start for a second, then move: the arm follows the script.
        for _ in 0..30 {
            sim.tick(&hub, 0.1);
        }
        let frames = hub.drain();
        assert!(frames.iter().all(|f| f.frame.forearm.is_some()));
        assert!(sim.arm.humerus.is_some(), "the scripted shoulder is driven");
        assert!(
            sim.arm.pronation_deg != 0.0,
            "forearm rotation is simulated"
        );

        // The sensor dropout scenario goes silent mid-movement and then recovers.
        sim.start_scenario(Scenario::SensorDropout, config);
        let mut counts = Vec::new();
        for _ in 0..80 {
            sim.tick(&hub, 0.1);
            counts.push(hub.drain().len());
        }
        assert!(counts.contains(&0), "packets stop: {counts:?}");
        assert!(
            counts.last().copied().unwrap_or(0) > 0 || counts.iter().rev().take(3).any(|c| *c > 0)
        );

        // Forearm dropout sends upper-arm-only packets.
        sim.start_scenario(Scenario::ForearmDropout, config);
        let mut only_upper = false;
        for _ in 0..80 {
            sim.tick(&hub, 0.1);
            only_upper |= hub
                .drain()
                .iter()
                .any(|f| f.frame.upper.is_some() && f.frame.forearm.is_none());
        }
        assert!(
            only_upper,
            "forearm packets stop while the upper arm continues"
        );

        // A noisy link produces valid but jittery packets and some invalid ones.
        sim.start_scenario(Scenario::NoisyLink, config);
        for _ in 0..400 {
            sim.tick(&hub, 0.05);
        }
        let stats = hub.stats();
        assert!(stats.invalid_packets > 0, "corrupt packets are counted");
        assert!(stats.packets > stats.invalid_packets);

        // Presets and stopping hand control back to the sliders and clear the noise.
        sim.rest_pose();
        assert_eq!(sim.input, SimInput::Sliders);
        assert_eq!(sim.arm.pronation_deg, 0.0);
        assert!(sim.scenario.is_none(), "presets end the script");
    }

    #[test]
    fn switching_arms_keeps_the_forearm_rotation() {
        let mut sim = SimDriver::new(Side::Right);
        sim.arm.pronation_deg = 33.0;
        sim.set_side(Side::Left);
        assert_eq!(sim.arm.pronation_deg, 33.0);
    }
}
