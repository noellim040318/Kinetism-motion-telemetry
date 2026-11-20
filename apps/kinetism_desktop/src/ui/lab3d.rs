//! 3D arm view and 3D fruit game.
//!
//! Drawn with egui's painter through a small perspective camera (no extra
//! engine). The wrist position comes from forward kinematics of the two IMU
//! orientations. It is not a measured hand position: there is no hand sensor,
//! no trunk sensor and no position tracking. The hand and blade follow the
//! forearm and are labelled as inferred.
//!
//! Nothing played here is saved: 3D rounds are practice results for the
//! physiotherapist and do not touch measurement records, MMT grades or the
//! Firestore schemas.

use crate::app::{KinetismApp, Screen};
use crate::arm3d::{ArmSkeleton, OrbitCamera, neutral_skeleton, skeleton_with_hand};
use crate::armpose::elevation_and_plane;
use crate::clinical::Side;
use crate::forearm::{ForearmRotation, RotationDirection, Unreliable};
use crate::game::Stage;
use crate::math::Vec3;
use crate::pnf3d::{
    Assistance, Difficulty, ElbowMode, Frame, Phase, PnfConfig, PnfEvent, PnfPattern, RotationRule,
    Round3d, Trajectory,
};
use crate::scenarios::Scenario;
use crate::sensor::Source;
use crate::theme;
use crate::tracker::CalStep;
use crate::ui::limb3d::{self, Face};
use crate::ui::{self, card};
use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, Vec2,
};
use std::collections::VecDeque;

const FLOOR_Z: f64 = -1.25;
const TRAIL_POINTS: usize = 28;
const FEEDBACK_SECONDS: f64 = 3.5;
/// Time constant of the display smoothing: small enough to stay responsive.
const SMOOTHING_SECONDS: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DemoStep {
    Rest,
    Forward,
    Neutral,
    Playing,
}

/// Realistic arm, or skeleton with joints, axes and frames for finding coordinate errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Anatomical,
    Debug,
}

/// What the shared 3D scene draws besides the arm.
pub(crate) struct SceneOpts<'a> {
    pub round: Option<&'a Round3d>,
    pub show_reference: bool,
    pub view: ViewMode,
    /// Pattern the debug view compares the arm against.
    pub config: PnfConfig,
    pub hud: bool,
}

pub struct LabState {
    pub camera: OrbitCamera,
    pub view: ViewMode,
    pub config: PnfConfig,
    pub round: Option<Round3d>,
    paused: bool,
    show_reference: bool,
    scenario: Scenario,
    demo: Option<DemoStep>,
    demo_timer: f64,
    /// Smoothed segment directions for drawing only (never used for scoring).
    shown: Option<(Vec3, Option<Vec3>, Vec3)>,
    trail: VecDeque<Vec3>,
    feedback: Vec<String>,
    feedback_left: f64,
    history: VecDeque<String>,
    clock: f64,
}

impl LabState {
    pub fn new(side: Side) -> Self {
        Self {
            camera: OrbitCamera::default(),
            view: ViewMode::Anatomical,
            config: PnfConfig::new(PnfPattern::D2Flexion, side),
            round: None,
            paused: false,
            show_reference: true,
            scenario: Scenario::D2Flexion,
            demo: None,
            demo_timer: 0.0,
            shown: None,
            trail: VecDeque::new(),
            feedback: Vec::new(),
            feedback_left: 0.0,
            history: VecDeque::new(),
            clock: 0.0,
        }
    }

    /// Demo options (`KINETISM_LAB_OPTS=scenario=D1Flexion,camera=top,view=debug,side=left`):
    /// for screenshots and demos. Returns the requested arm side, if any.
    pub fn apply_demo_options(&mut self, options: &str) -> Option<Side> {
        let mut side = None;
        for part in options.split(',') {
            let Some((key, value)) = part.split_once('=') else {
                continue;
            };
            let value = value.trim();
            match key.trim() {
                "scenario" => {
                    if let Some(s) = Scenario::ALL
                        .into_iter()
                        .find(|s| format!("{s:?}").eq_ignore_ascii_case(value))
                    {
                        self.scenario = s;
                    }
                }
                "camera" => {
                    let (yaw, pitch) = match value {
                        "front" => (std::f64::consts::PI, 0.0),
                        "side" => (std::f64::consts::FRAC_PI_2, 0.0),
                        "top" => (0.0, 1.4),
                        "behind" => (0.0, 0.0),
                        _ => continue,
                    };
                    self.camera = OrbitCamera {
                        yaw,
                        pitch,
                        ..OrbitCamera::default()
                    };
                }
                "view" => {
                    self.view = if value == "debug" {
                        ViewMode::Debug
                    } else {
                        ViewMode::Anatomical
                    };
                }
                "side" => {
                    side = match value {
                        "left" => Some(Side::Left),
                        "right" => Some(Side::Right),
                        _ => None,
                    };
                }
                _ => {}
            }
        }
        side
    }

    /// Blade trail: the newest blade tip, or cleared when tracking is lost.
    pub fn push_trail(&mut self, tip: Option<Vec3>) {
        match tip {
            Some(p) => {
                self.trail.push_back(p);
                while self.trail.len() > TRAIL_POINTS {
                    self.trail.pop_front();
                }
            }
            None => self.trail.clear(),
        }
    }

    /// Supportive on-screen feedback for a round event.
    pub fn round_feedback(&mut self, event: &PnfEvent) {
        match event {
            PnfEvent::RepScored(rep) => {
                self.say(rep.feedback().into_iter().map(String::from).collect());
            }
            PnfEvent::RepDiscarded => self.say(vec![
                "Movement tracking unavailable. This rep was not counted".into(),
            ]),
            PnfEvent::Finished => self.say(vec!["Round complete".into()]),
            _ => {}
        }
    }

    fn say(&mut self, lines: Vec<String>) {
        self.feedback = lines;
        self.feedback_left = FEEDBACK_SECONDS;
    }

    fn note(&mut self, line: String) {
        self.history.push_front(line);
        self.history.truncate(6);
    }
}

/// Which way the palm faces in the body frame (inferred from the forearm).
fn limb3d_palm(sk: &ArmSkeleton) -> &'static str {
    if sk.palm_normal.z > 0.6 {
        "up"
    } else if sk.palm_normal.z < -0.6 {
        "down"
    } else {
        "sideways"
    }
}

/// Signed whole degrees, never "-0".
fn signed_deg(v: f64) -> String {
    format!("{:+.0}", if v.abs() < 0.5 { 0.0 } else { v })
}

fn rgba(c: Color32, a: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (a.clamp(0.0, 1.0) * 255.0) as u8)
}

/// One projected scene: converts body-frame points to screen pixels.
struct Scene {
    camera: OrbitCamera,
    rect: Rect,
}

struct Pt3 {
    pos: Pos2,
    /// Pixels per arm length at this depth.
    scale: f32,
    depth: f64,
}

impl Scene {
    fn project(&self, p: Vec3) -> Option<Pt3> {
        let h = f64::from(self.rect.height());
        let q = self.camera.project(p, h)?;
        let focal = (h * 0.5) / (self.camera.fov_deg.to_radians() * 0.5).tan();
        Some(Pt3 {
            pos: self.rect.center() + Vec2::new(q.x as f32, q.y as f32),
            scale: (focal / q.depth) as f32,
            depth: q.depth,
        })
    }

    /// Fade with distance for a depth cue.
    fn fade(&self, depth: f64) -> f32 {
        let near = (self.camera.distance - 1.5).max(0.5);
        (1.0 - ((depth - near) / 4.0).clamp(0.0, 0.55)) as f32
    }

    fn line(&self, painter: &egui::Painter, a: Vec3, b: Vec3, width: f32, color: Color32) {
        if let (Some(pa), Some(pb)) = (self.project(a), self.project(b)) {
            let fade = self.fade((pa.depth + pb.depth) * 0.5);
            let w = (width * (pa.scale + pb.scale) * 0.5 / 220.0).clamp(1.0, 40.0);
            painter.line_segment(
                [pa.pos, pb.pos],
                Stroke::new(w, rgba(color, color.a() as f32 / 255.0 * fade)),
            );
        }
    }

    /// Faint drop line and shadow on the floor: the main depth cue.
    fn drop_shadow(&self, painter: &egui::Painter, p: Vec3, radius: f64) {
        let floor = Vec3::new(p.x, p.y, FLOOR_Z);
        self.line(painter, p, floor, 2.0, Color32::from_white_alpha(28));
        if let Some(q) = self.project(floor) {
            let r = radius as f32 * q.scale;
            painter.add(Shape::ellipse_filled(
                q.pos,
                Vec2::new(r, r * 0.4),
                Color32::from_black_alpha(80),
            ));
        }
    }
}

fn draw_floor_and_body(scene: &Scene, painter: &egui::Painter, side: Side) {
    let grid = Color32::from_white_alpha(22);
    for i in -4..=4 {
        let v = f64::from(i) * 0.5;
        scene.line(
            painter,
            Vec3::new(v, -2.0, FLOOR_Z),
            Vec3::new(v, 2.0, FLOOR_Z),
            2.0,
            grid,
        );
        scene.line(
            painter,
            Vec3::new(-2.0, v, FLOOR_Z),
            Vec3::new(2.0, v, FLOOR_Z),
            2.0,
            grid,
        );
    }
    // Ghost trunk (fixed: there is no trunk sensor).
    let body = Color32::from_white_alpha(46);
    let other = Vec3::new(-side.sign() * 0.65, 0.0, 0.0);
    let hip_a = Vec3::new(0.0, 0.0, -0.95);
    let hip_b = other + Vec3::new(0.0, 0.0, -0.95);
    for (a, b) in [
        (Vec3::ZERO, other),
        (other, hip_b),
        (hip_b, hip_a),
        (hip_a, Vec3::ZERO),
    ] {
        scene.line(painter, a, b, 5.0, body);
    }
    let neck = other.scale(0.5);
    scene.line(painter, neck, neck + Vec3::new(0.0, 0.0, 0.22), 5.0, body);
    if let Some(head) = scene.project(neck + Vec3::new(0.0, 0.0, 0.40)) {
        painter.circle_stroke(head.pos, 0.18 * head.scale, Stroke::new(2.0, body));
    }
    let facing = neck + Vec3::new(0.0, 0.0, 0.40);
    scene.line(
        painter,
        facing,
        facing + Vec3::new(0.0, 0.2, 0.0),
        3.0,
        body,
    );
}

/// Light direction (towards the light) in camera terms: upper left, towards the viewer.
fn light_direction(camera: &OrbitCamera) -> Vec3 {
    let (right, up, forward) = camera.basis();
    (up.scale(0.7) - right.scale(0.45) - forward.scale(0.6))
        .normalized()
        .unwrap_or(Vec3::Z)
}

/// Draw mesh faces far to near, culling the ones facing away and shading the rest.
/// `ghost` draws a flat translucent silhouette (the neutral reference).
fn draw_faces(scene: &Scene, painter: &egui::Painter, faces: &[Face], alpha: f32, ghost: bool) {
    let eye = scene.camera.position();
    let light = light_direction(&scene.camera);
    let mut drawn: Vec<(f64, Vec<Pos2>, Color32)> = Vec::with_capacity(faces.len());
    for face in faces {
        if (ghost && face.decal) || !face.front_facing(eye) {
            continue;
        }
        let projected: Vec<Pt3> = face.v.iter().filter_map(|p| scene.project(*p)).collect();
        if projected.len() != 4 {
            continue;
        }
        let depth = projected.iter().map(|q| q.depth).sum::<f64>() / 4.0;
        let colour = if ghost {
            Color32::from_rgba_unmultiplied(190, 205, 215, (alpha * 90.0) as u8)
        } else {
            let lit = 0.38 + 0.62 * face.n.dot(light).max(0.0) as f32;
            let fade = scene.fade(depth);
            let c = face.color;
            Color32::from_rgba_unmultiplied(
                (f32::from(c.r()) * lit).min(255.0) as u8,
                (f32::from(c.g()) * lit).min(255.0) as u8,
                (f32::from(c.b()) * lit).min(255.0) as u8,
                (alpha * fade * 255.0) as u8,
            )
        };
        // Markings sit on the hand surface: draw them after what is under them.
        let order = if face.decal { depth - 0.05 } else { depth };
        // Triangles repeat a vertex: a zero-length edge makes the stroke spike.
        let mut points: Vec<Pos2> = Vec::with_capacity(4);
        for q in &projected {
            if points
                .last()
                .is_none_or(|p| (p.x - q.pos.x).abs() > 0.01 || (p.y - q.pos.y).abs() > 0.01)
            {
                points.push(q.pos);
            }
        }
        if points.len() > 1 && points[0].distance(points[points.len() - 1]) < 0.01 {
            points.pop();
        }
        if points.len() < 3 {
            continue;
        }
        // Edge-on faces cover nothing and their sharp corners make egui's
        // outline spike, so skip them. Growing each face a little closes the
        // hairline seams between neighbours without drawing outlines.
        let centre = points.iter().fold(Vec2::ZERO, |a, p| a + p.to_vec2()) / points.len() as f32;
        let radius = points
            .iter()
            .map(|p| (p.to_vec2() - centre).length())
            .sum::<f32>()
            / points.len() as f32;
        let twice_area: f32 = (0..points.len())
            .map(|i| {
                let (a, b) = (points[i], points[(i + 1) % points.len()]);
                a.x * b.y - b.x * a.y
            })
            .sum::<f32>()
            .abs();
        if radius < 0.4 || twice_area < 1.0 {
            continue;
        }
        let grow = 1.0 + 0.6 / radius;
        for p in &mut points {
            *p = (centre + (p.to_vec2() - centre) * grow).to_pos2();
        }
        drawn.push((order, points, colour));
    }
    drawn.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, points, colour) in drawn {
        painter.add(Shape::convex_polygon(points, colour, Stroke::NONE));
    }
}

fn draw_arm(
    scene: &Scene,
    painter: &egui::Painter,
    sk: &ArmSkeleton,
    alpha: f32,
    ghost: bool,
    detailed: bool,
) {
    draw_faces(scene, painter, &limb3d::arm_faces(sk), alpha, ghost);
    if detailed {
        // Blade: glowing line from the fingertips to the (unchanged) blade tip.
        let root = sk.wrist + sk.forearm_dir.scale(limb3d::HAND_VISUAL_LENGTH);
        let colour = |col: Color32, a: f32| rgba(col, a * alpha);
        scene.line(
            painter,
            root,
            sk.blade_tip,
            14.0,
            colour(Color32::from_rgb(0x9F, 0xE8, 0xFF), 0.30),
        );
        scene.line(
            painter,
            root,
            sk.blade_tip,
            5.0,
            colour(Color32::WHITE, 1.0),
        );
    }
}

/// Dial showing the estimated forearm rotation.
fn rotation_dial(
    painter: &egui::Painter,
    centre: Pos2,
    radius: f32,
    rot: Option<ForearmRotation>,
    side: Side,
) {
    painter.circle_filled(centre, radius, Color32::from_black_alpha(120));
    painter.circle_stroke(centre, radius, Stroke::new(2.0, theme::FAINT));
    // Neutral at the top; pronation turns toward the patient's medial side.
    painter.line_segment(
        [
            centre + Vec2::new(0.0, -radius),
            centre + Vec2::new(0.0, -radius * 0.78),
        ],
        Stroke::new(2.0, theme::MUTED),
    );
    if let Some(r) = rot {
        // Mirror view: a pronating right arm turns the needle to the screen's left (medial).
        let turn = (-side.sign() * r.angle_deg).to_radians() as f32;
        let tip = centre + Vec2::new(turn.sin(), -turn.cos()) * radius * 0.82;
        let colour = match r.direction {
            RotationDirection::Neutral => theme::GOOD,
            _ => theme::ACCENT,
        };
        painter.line_segment([centre, tip], Stroke::new(4.0, colour));
        painter.circle_filled(tip, 5.0, colour);
    }
    painter.circle_filled(centre, 4.0, theme::TEXT);
}

impl KinetismApp {
    /// Per-frame data work for the 3D screen: demo script, display smoothing
    /// and the round.
    pub(crate) fn update_lab(&mut self, dt: f64, resumed: bool) {
        self.lab.clock += dt;
        self.lab.feedback_left = (self.lab.feedback_left - dt).max(0.0);
        self.advance_lab_demo(dt);

        // Smooth the drawn arm toward the newest pose (the round uses raw data).
        if let (Some(pose), Some(sk)) = (self.tracker.pose, self.tracker.skeleton()) {
            let alpha = 1.0 - (-dt / SMOOTHING_SECONDS).exp();
            let blend =
                |old: Vec3, new: Vec3| (old + (new - old).scale(alpha)).normalized().unwrap_or(new);
            self.lab.shown = Some(match self.lab.shown {
                Some((u, f, p)) => (
                    blend(u, pose.upper_dir),
                    match (f, pose.forearm_dir) {
                        (Some(f), Some(n)) => Some(blend(f, n)),
                        (_, n) => n,
                    },
                    blend(p, sk.palm_normal),
                ),
                None => (pose.upper_dir, pose.forearm_dir, sk.palm_normal),
            });
        }
        if self.screen != Screen::Lab3d {
            return;
        }
        let tracked =
            self.tracker.is_calibrated() && self.tracker.has_fresh_data() && !self.lab.paused;
        let rotation_needed = self.lab.config.rotation.is_some();
        let frame = tracked
            .then(|| self.tracker.skeleton())
            .flatten()
            .and_then(|sk| {
                let rotation = rotation_needed
                    .then(|| self.tracker.reliable_rotation_deg())
                    .flatten();
                Frame::from_skeleton(&sk, rotation)
            });
        if let Some(f) = frame {
            self.lab.trail.push_back(f.blade_tip);
            while self.lab.trail.len() > TRAIL_POINTS {
                self.lab.trail.pop_front();
            }
        } else {
            self.lab.trail.clear();
        }
        let events = {
            let Some(round) = self.lab.round.as_mut() else {
                return;
            };
            if self.lab.paused {
                return;
            }
            if resumed {
                round.interrupt();
            }
            round.update(frame, dt)
        };
        for event in events {
            match event {
                PnfEvent::RepStarted(n) => self.lab.note(format!("Rep {n} started")),
                PnfEvent::Sliced { .. } | PnfEvent::ReachedTop => {}
                PnfEvent::RepScored(rep) => {
                    let lines: Vec<String> = rep.feedback().into_iter().map(String::from).collect();
                    let rotation = rep.peak_rotation_deg.map_or(String::new(), |d| {
                        format!(", forearm rotation up to {d:.0}°")
                    });
                    self.lab.note(format!(
                        "Fruit {}/{}, path {:.0}% in tolerance{rotation}",
                        rep.fruits_sliced,
                        rep.fruits_total,
                        rep.within_tolerance * 100.0
                    ));
                    self.lab.say(lines);
                }
                PnfEvent::RepDiscarded => {
                    self.lab
                        .note("Rep not counted: tracking was interrupted".into());
                    self.lab.say(vec![
                        "Movement tracking unavailable. This rep was not counted".into(),
                    ]);
                }
                PnfEvent::TrackingLost => {}
                PnfEvent::Finished => self.lab.say(vec!["Round complete".into()]),
            }
        }
    }

    /// Calibrate the virtual arm and loop the chosen scenario. Everything it
    /// produces is simulated and never saved.
    pub(crate) fn run_lab_demo(&mut self) {
        if !self.simulated() {
            self.switch_source(Source::Simulator);
        }
        self.sim.rest_pose();
        self.lab.round = None;
        self.lab.config.side = self.side();
        // Walk the diagonal the scenario was built for; the combined
        // scenario includes forearm supination, so assess it.
        self.lab.config.pattern = self.lab.scenario.target_pattern();
        if matches!(
            self.lab.scenario,
            Scenario::D1Flexion
                | Scenario::D1Extension
                | Scenario::D2Flexion
                | Scenario::D2Extension
        ) && self.lab.config.rotation.is_none()
        {
            self.lab.config = self.lab.config.with_suggested_rotation();
        }
        self.lab.demo = Some(DemoStep::Rest);
        self.lab.demo_timer = 0.0;
        self.tracker.recalibrate();
    }

    fn advance_lab_demo(&mut self, dt: f64) {
        let Some(step) = self.lab.demo else {
            return;
        };
        if !self.simulated() {
            self.lab.demo = None;
            return;
        }
        self.lab.demo_timer += dt;
        match step {
            DemoStep::Rest => match self.tracker.step {
                CalStep::Rest => {
                    self.sim.rest_pose();
                    if self.lab.demo_timer > 0.4 {
                        self.tracker.start_rest_capture();
                    }
                }
                CalStep::Forward | CalStep::CapturingForward => {
                    self.lab.demo = Some(DemoStep::Forward);
                    self.lab.demo_timer = 0.0;
                }
                _ => {}
            },
            DemoStep::Forward => {
                self.sim.forward_pose();
                if self.lab.demo_timer > 1.5 && self.tracker.step == CalStep::Forward {
                    self.tracker.start_forward_capture();
                }
                if self.tracker.is_calibrated() {
                    self.lab.demo = Some(DemoStep::Neutral);
                    self.lab.demo_timer = 0.0;
                }
            }
            DemoStep::Neutral => {
                self.sim.forward_pose();
                if self.lab.demo_timer > 0.8
                    && !self.tracker.is_capturing_neutral()
                    && !self.tracker.has_rotation_calibration()
                {
                    self.tracker.start_neutral_capture();
                }
                if self.tracker.has_rotation_calibration() {
                    self.sim.start_scenario(self.lab.scenario, self.lab.config);
                    self.lab.demo = Some(DemoStep::Playing);
                    self.lab.round = Some(Round3d::new(self.lab.config));
                    self.lab.history.clear();
                }
            }
            DemoStep::Playing => {}
        }
    }

    fn start_lab_round(&mut self) {
        if !self.tracker.is_calibrated() {
            self.toast("Recalibration needed: calibrate the arm first.", true);
            return;
        }
        if self.lab.config.rotation.is_some() && !self.tracker.has_rotation_calibration() {
            self.toast(
                "Set the forearm neutral position first, or switch rotation assessment off.",
                true,
            );
            return;
        }
        self.lab.config = self.lab.config.sanitized();
        self.lab.round = Some(Round3d::new(self.lab.config));
        self.lab.history.clear();
        self.lab.paused = false;
    }

    /// Message that explains why nothing is being scored right now.
    fn lab_tracking_message(&self, rotation_needed: bool) -> Option<&'static str> {
        if !self.tracker.is_calibrated() {
            return Some("Recalibration needed");
        }
        if !self.tracker.has_fresh_data() {
            return Some("Movement tracking unavailable");
        }
        if self.tracker.pose.is_some_and(|p| p.forearm_dir.is_none()) {
            return Some("Movement tracking unavailable: forearm sensor not reporting");
        }
        if rotation_needed && self.tracker.reliable_rotation_deg().is_none() {
            return Some("Forearm rotation unavailable");
        }
        None
    }

    pub(crate) fn lab_screen(&mut self, ui: &mut egui::Ui) {
        ui::eyebrow(ui, "3D arm and Fruit PNF");
        ui.horizontal_wrapped(|ui| {
            ui::title(ui, "3D arm view");
            ui.add_space(12.0);
            if self.simulated() {
                ui::chip(ui, "SIMULATED: practice data, not saved", theme::WARN);
            }
            ui::chip(ui, "Results are not saved", theme::MUTED);
        });
        ui.add_space(4.0);
        let left_width = 330.0_f32.min(ui.available_width() * 0.4);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                Vec2::new(left_width, ui.available_height()),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("lab-controls")
                        .auto_shrink([false; 2])
                        .show(ui, |ui| self.lab_controls(ui));
                },
            );
            ui.add_space(8.0);
            self.lab_canvas(ui);
        });
    }

    fn lab_controls(&mut self, ui: &mut egui::Ui) {
        // 1. Calibration
        card(ui, |ui| {
            ui.label(RichText::new("1 · Arm calibration").strong());
            if self.tracker.is_calibrated() {
                ui.label(RichText::new("✓ Calibrated").color(theme::GOOD));
                if ui.small_button("Recalibrate").clicked() {
                    self.tracker.recalibrate();
                    self.lab.round = None;
                }
            } else {
                self.calibration_panel(ui);
            }
        });
        ui.add_space(6.0);
        // 2. Forearm neutral
        card(ui, |ui| {
            ui.label(RichText::new("2 · Forearm rotation neutral").strong());
            ui.label(ui::small(
                "Elbow straight, thumb forward, hold still. Needed only to assess forearm rotation.",
            ));
            let can = self.tracker.is_calibrated() && !self.tracker.forearm_missing();
            if self.tracker.is_capturing_neutral() {
                ui.add(
                    egui::ProgressBar::new(self.tracker.neutral_progress())
                        .text("Hold still…")
                        .fill(theme::ACCENT),
                );
            } else if ui::primary_enabled(
                ui,
                can,
                if self.tracker.has_rotation_calibration() {
                    "Set neutral again"
                } else {
                    "Set forearm neutral"
                },
            )
            .clicked()
            {
                self.tracker.start_neutral_capture();
            }
            match self.tracker.forearm_rotation {
                Some(Ok(r)) => {
                    ui.label(format!(
                        "Forearm Rotation: {}° ({})",
                        signed_deg(r.angle_deg),
                        r.direction.label()
                    ));
                }
                Some(Err(e)) => {
                    ui.label(RichText::new(e.label()).color(theme::WARN));
                }
                None if self.tracker.is_calibrated() => {
                    ui.label(ui::small(Unreliable::NotCalibrated.label()));
                }
                None => {}
            }
            ui.label(ui::small(
                "No magnetometer: the sensors' relative yaw slowly drifts and looks like forearm rotation. Set the neutral again if it no longer reads about 0 at neutral.",
            ));
        });
        ui.add_space(6.0);
        // 3. Therapist configuration
        card(ui, |ui| self.lab_config_panel(ui));
        ui.add_space(6.0);
        // 4. Round
        card(ui, |ui| {
            ui.label(RichText::new("4 · Round").strong());
            ui.horizontal(|ui| {
                if ui::primary(
                    ui,
                    if self.lab.round.is_some() {
                        "Restart round"
                    } else {
                        "Start round"
                    },
                )
                .clicked()
                {
                    self.start_lab_round();
                }
                if self.lab.round.is_some() {
                    if ui
                        .button(if self.lab.paused { "Resume" } else { "Pause" })
                        .clicked()
                    {
                        self.lab.paused = !self.lab.paused;
                    }
                    if ui.button("Stop").clicked() {
                        self.lab.round = None;
                    }
                }
            });
            if let Some(round) = &self.lab.round {
                ui.label(format!(
                    "Rep {}/{} · spatial score {:.0}% · {} pts",
                    round.scored_reps().min(round.config.reps),
                    round.config.reps,
                    round.spatial_fraction() * 100.0,
                    round.points
                ));
                if let Some(rate) = round.rotation_rate() {
                    ui.label(format!(
                        "Forearm rotation reached in {:.0}% of reps",
                        rate * 100.0
                    ));
                }
                if round.discarded > 0 {
                    ui.label(ui::small(format!(
                        "{} rep(s) not counted (tracking interrupted)",
                        round.discarded
                    )));
                }
                ui.label(ui::small(
                    "Game results are practice feedback. They are not MMT grades.",
                ));
            }
            for line in &self.lab.history {
                ui.label(ui::small(line.as_str()));
            }
        });
        ui.add_space(6.0);
        // 5. Simulator demo
        card(ui, |ui| {
            ui.label(RichText::new("Simulated demo (no hardware)").strong());
            ui.label(ui::small(
                "Drives the simulator through calibration and a scripted movement. Simulated data is never saved.",
            ));
            egui::ComboBox::from_id_salt("lab-scenario")
                .selected_text(self.lab.scenario.label())
                .width(260.0)
                .show_ui(ui, |ui| {
                    for s in Scenario::ALL {
                        ui.selectable_value(&mut self.lab.scenario, s, s.label());
                    }
                });
            ui.horizontal(|ui| {
                if ui::primary(ui, "Run demo").clicked() {
                    self.run_lab_demo();
                }
                if self.lab.demo.is_some() && ui.button("Stop demo").clicked() {
                    self.lab.demo = None;
                    self.sim.stop_scenario();
                }
            });
            match self.lab.demo {
                Some(DemoStep::Rest | DemoStep::Forward) => {
                    ui.label(ui::small("Calibrating the virtual arm…"));
                }
                Some(DemoStep::Neutral) => {
                    ui.label(ui::small("Capturing the forearm neutral…"));
                }
                Some(DemoStep::Playing) => {
                    ui.label(ui::small(format!(
                        "Playing: {} (repeats)",
                        self.lab.scenario.label()
                    )));
                }
                None => {}
            }
        });
        ui.add_space(6.0);
        card(ui, |ui| {
            ui.label(RichText::new("What this view can and cannot show").strong());
            for line in [
                "Wrist, hand and blade positions come from the two segment orientations and fixed segment lengths. Absolute hand position is not measured.",
                "The hand and blade follow the forearm. Their orientation is inferred, not measured wrist articulation.",
                "There is no trunk sensor: the shoulder is fixed in space and trunk lean is invisible.",
                "Forearm Rotation is the twist of the forearm relative to the upper arm, not wrist flexion, extension or deviation.",
            ] {
                ui.label(ui::small(format!("• {line}")));
            }
        });
    }

    fn lab_config_panel(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("3 · PNF pattern (therapist settings)").strong());
        ui.label(ui::small(
            "Draft templates for physiotherapist review, not prescriptions. Adjust to the patient's range.",
        ));
        let side = self.side();
        let mut changed_side = None;
        ui.horizontal(|ui| {
            ui.label("Affected arm:");
            let mut s = side;
            ui.selectable_value(&mut s, Side::Left, "Left");
            ui.selectable_value(&mut s, Side::Right, "Right");
            if s != side {
                changed_side = Some(s);
            }
        });
        if let Some(s) = changed_side {
            self.set_side(s);
            self.lab.round = None;
        }
        self.lab.config.side = self.side();
        let cfg = &mut self.lab.config;
        let before = cfg.pattern;
        egui::ComboBox::from_id_salt("lab-pattern")
            .selected_text(cfg.pattern.label())
            .show_ui(ui, |ui| {
                for p in PnfPattern::ALL {
                    ui.selectable_value(&mut cfg.pattern, p, p.label());
                }
            });
        if cfg.pattern != before
            && let Some(rule) = &mut cfg.rotation
        {
            rule.expected = cfg.pattern.suggested_rotation();
        }
        ui.add(egui::Slider::new(&mut cfg.reach, 0.4..=1.0).text("Reach (of path)"));
        ui.add(egui::Slider::new(&mut cfg.tolerance, 0.05..=0.4).text("Path tolerance"));
        ui.add(
            egui::Slider::new(&mut cfg.start_shift_deg[0], -30.0..=30.0)
                .text("Start shift: plane °"),
        );
        ui.add(
            egui::Slider::new(&mut cfg.start_shift_deg[1], -30.0..=30.0)
                .text("Start shift: elevation °"),
        );
        egui::ComboBox::from_id_salt("lab-elbow")
            .selected_text(format!("Elbow: {}", cfg.elbow_mode.label()))
            .show_ui(ui, |ui| {
                for m in ElbowMode::ALL {
                    ui.selectable_value(&mut cfg.elbow_mode, m, m.label());
                }
            });
        if cfg.elbow_mode == ElbowMode::Straight {
            ui.add(egui::Slider::new(&mut cfg.elbow_deg, 0.0..=90.0).text("Held elbow flexion °"));
        } else {
            ui.label(ui::small(
                "Example elbow trajectory for this pattern (unvalidated, for physio review).",
            ));
        }
        ui.add(egui::Slider::new(&mut cfg.reps, 1..=20).text("Reps"));
        ui.horizontal(|ui| {
            ui.label("Difficulty:");
            for d in Difficulty::ALL {
                ui.selectable_value(&mut cfg.difficulty, d, d.label());
            }
        });
        egui::ComboBox::from_id_salt("lab-assist")
            .selected_text(format!("Assistance: {}", cfg.assistance.label()))
            .show_ui(ui, |ui| {
                for a in Assistance::ALL {
                    ui.selectable_value(&mut cfg.assistance, a, a.label());
                }
            });
        let mut assess = cfg.rotation.is_some();
        if ui
            .checkbox(&mut assess, "Assess forearm rotation")
            .changed()
        {
            cfg.rotation = assess.then(|| {
                PnfConfig::new(cfg.pattern, cfg.side)
                    .with_suggested_rotation()
                    .rotation
                    .expect("rule")
            });
        }
        if let Some(rule) = &mut cfg.rotation {
            rotation_rule_controls(ui, rule);
        }
        *cfg = cfg.sanitized();
    }

    /// The smoothed arm as drawn, with the palm from the forearm's orientation.
    fn shown_skeleton(&self, side: Side) -> Option<ArmSkeleton> {
        self.lab
            .shown
            .map(|(u, f, palm)| skeleton_with_hand(u, f, palm, side))
    }

    /// Orbit, zoom and reset input for a 3D canvas.
    pub(crate) fn scene_input(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.dragged() {
            let d = response.drag_delta();
            self.lab
                .camera
                .orbit(-f64::from(d.x) * 0.01, f64::from(d.y) * 0.01);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.lab.camera.zoom(f64::from(scroll) * 0.002);
            }
        }
        if response.double_clicked() {
            self.lab.camera = OrbitCamera::default();
        }
    }

    /// Camera presets (and, in the Lab, the reference and view toggles).
    pub(crate) fn view_controls(&mut self, ui: &mut egui::Ui, rect: Rect, lab: bool) {
        let width = if lab { 580.0 } else { 300.0 };
        let controls = Rect::from_min_size(
            rect.right_top() + Vec2::new(-width - 10.0, 10.0),
            Vec2::new(width, 30.0),
        );
        let mut preset: Option<OrbitCamera> = None;
        ui.scope_builder(egui::UiBuilder::new().max_rect(controls), |ui| {
            ui.horizontal(|ui| {
                if ui.small_button("Front").clicked() {
                    preset = Some(OrbitCamera {
                        yaw: std::f64::consts::PI,
                        pitch: 0.0,
                        ..OrbitCamera::default()
                    });
                }
                if ui.small_button("Behind").clicked() {
                    preset = Some(OrbitCamera::behind());
                }
                if ui.small_button("Side").clicked() {
                    preset = Some(OrbitCamera {
                        yaw: std::f64::consts::FRAC_PI_2,
                        pitch: 0.0,
                        ..OrbitCamera::default()
                    });
                }
                if ui.small_button("Top").clicked() {
                    preset = Some(OrbitCamera {
                        yaw: 0.0,
                        pitch: 1.4,
                        ..OrbitCamera::default()
                    });
                }
                if lab {
                    ui.checkbox(&mut self.lab.show_reference, "Neutral");
                    ui.selectable_value(&mut self.lab.view, ViewMode::Anatomical, "Anatomical");
                    ui.selectable_value(&mut self.lab.view, ViewMode::Debug, "Debug");
                }
            });
        });
        if let Some(cam) = preset {
            self.lab.camera = cam;
        }
    }

    /// The real Fruit PNF game (levels 1-2) in 3D: the same renderer, arm and
    /// kinematics as the Lab, with the game's countdown, score and result.
    pub(crate) fn game_canvas_3d(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        painter.rect_filled(rect, 14.0, Color32::from_rgb(0x0E, 0x18, 0x1E));
        self.scene_input(ui, &response);
        let painter = painter.with_clip_rect(rect);
        let Some(game) = self.play.game.as_ref() else {
            return;
        };
        let crate::game::Mode::Pnf3d(round) = &game.mode else {
            return;
        };
        let (stage, points, score) = (game.stage, game.points(), game.score_percent());
        let (rep, reps, phase) = (round.scored_reps(), round.config.reps, round.phase);
        let (sliced, fruit_total) = (
            round.fruits.iter().filter(|f| f.sliced).count(),
            round.fruits.len(),
        );
        {
            let opts = SceneOpts {
                round: Some(round),
                show_reference: false,
                view: ViewMode::Anatomical,
                config: round.config,
                hud: true,
            };
            self.paint_scene(&painter, rect, self.lab.camera, &opts);
        }
        let hud = |text: String, at: Pos2, align: Align2, size: f32, color: Color32| {
            painter.text(
                at + Vec2::new(2.0, 2.0),
                align,
                &text,
                FontId::proportional(size),
                Color32::from_black_alpha(150),
            );
            painter.text(at, align, text, FontId::proportional(size), color);
        };
        hud(
            format!("{points}"),
            rect.right_top() + Vec2::new(-24.0, 56.0),
            Align2::RIGHT_TOP,
            40.0,
            Color32::from_rgb(0xFF, 0xD5, 0x4F),
        );
        hud(
            format!(
                "Rep {}/{}  ·  Fruit {sliced}/{fruit_total}",
                (rep + 1).min(reps),
                reps
            ),
            rect.left_top() + Vec2::new(22.0, 120.0),
            Align2::LEFT_TOP,
            20.0,
            theme::TEXT,
        );
        hud(
            format!("Score {score:.0}%"),
            rect.left_top() + Vec2::new(22.0, 146.0),
            Align2::LEFT_TOP,
            16.0,
            theme::MUTED,
        );
        if stage == Stage::Playing {
            let instruction = match phase {
                Phase::GoToStart => "Move your hand to the glowing Start ring",
                Phase::Swipe => "Slice the fruit along the diagonal up to the Target",
                Phase::Return => "Bring it back to Start",
                Phase::Done => "",
            };
            hud(
                instruction.into(),
                Pos2::new(rect.center().x, rect.bottom() - 26.0),
                Align2::CENTER_BOTTOM,
                20.0,
                Color32::WHITE,
            );
        }
        if self.lab.feedback_left > 0.0 {
            for (i, line) in self.lab.feedback.iter().enumerate() {
                hud(
                    line.clone(),
                    rect.center_top() + Vec2::new(0.0, 100.0 + i as f32 * 30.0),
                    Align2::CENTER_CENTER,
                    24.0,
                    Color32::WHITE,
                );
            }
        }
        if let Stage::Countdown(left) = stage {
            painter.rect_filled(rect, 14.0, Color32::from_black_alpha(90));
            hud(
                format!("{}", left.ceil() as u32),
                rect.center(),
                Align2::CENTER_CENTER,
                120.0,
                Color32::WHITE,
            );
            hud(
                "Get ready".into(),
                rect.center() + Vec2::new(0.0, 80.0),
                Align2::CENTER_CENTER,
                22.0,
                theme::TEXT,
            );
        }
        if self.play.paused {
            painter.rect_filled(rect, 14.0, Color32::from_black_alpha(140));
            hud(
                "Paused".into(),
                rect.center(),
                Align2::CENTER_CENTER,
                56.0,
                Color32::WHITE,
            );
        }
        if stage == Stage::Playing
            && let Some(msg) = self.lab_tracking_message(false)
        {
            hud(
                msg.into(),
                rect.center() - Vec2::new(0.0, 120.0),
                Align2::CENTER_CENTER,
                22.0,
                theme::WARN,
            );
        }
        self.view_controls(ui, rect, false);
        if stage != Stage::Finished && !self.tracker.has_rotation_calibration() && !self.simulated()
        {
            // Without the neutral the palm cannot follow the forearm. Real sensors need it set.
            let area = Rect::from_min_size(
                rect.right_bottom() + Vec2::new(-470.0, -62.0),
                Vec2::new(330.0, 30.0),
            );
            let mut set = false;
            ui.scope_builder(egui::UiBuilder::new().max_rect(area), |ui| {
                if self.tracker.is_capturing_neutral() {
                    ui.label(ui::small("Hold still: capturing the forearm neutral…"));
                } else if ui
                    .button("Set forearm neutral (elbow straight, thumb forward)")
                    .clicked()
                {
                    set = true;
                }
            });
            if set {
                self.tracker.start_neutral_capture();
            }
        }
        if stage == Stage::Finished {
            self.result_overlay(ui, rect);
        }
    }

    fn lab_canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        painter.rect_filled(rect, 14.0, Color32::from_rgb(0x0E, 0x18, 0x1E));
        self.scene_input(ui, &response);
        let painter = painter.with_clip_rect(rect);
        {
            let opts = SceneOpts {
                round: self.lab.round.as_ref(),
                show_reference: self.lab.show_reference,
                view: self.lab.view,
                config: self.lab.config,
                hud: true,
            };
            self.paint_scene(&painter, rect, self.lab.camera, &opts);
        }
        if self.lab.feedback_left > 0.0 {
            for (i, line) in self.lab.feedback.iter().enumerate() {
                self.canvas_text(
                    &painter,
                    rect.center_top() + Vec2::new(0.0, 100.0 + i as f32 * 30.0),
                    line,
                    Color32::WHITE,
                    24.0,
                );
            }
        }
        if let Some(msg) = self
            .lab_tracking_message(self.lab.config.rotation.is_some())
            .filter(|_| self.lab.round.is_some() || self.tracker.pose.is_some())
        {
            self.canvas_text(
                &painter,
                rect.center_bottom() - Vec2::new(0.0, 40.0),
                msg,
                theme::WARN,
                20.0,
            );
        }
        if self.lab.paused {
            self.canvas_text(&painter, rect.center(), "Paused", Color32::WHITE, 48.0);
        }
        painter.text(
            rect.left_bottom() + Vec2::new(14.0, -12.0),
            Align2::LEFT_BOTTOM,
            "Drag: rotate · Scroll: zoom · Double-click: reset view",
            FontId::proportional(13.0),
            theme::FAINT,
        );
        self.view_controls(ui, rect, true);
    }

    /// The whole 3D scene: body, arm, path, fruit. Shared by the Lab, the game
    /// and the small inset on the 2D levels, so there is one renderer and one
    /// kinematics pipeline.
    pub(crate) fn paint_scene(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        camera: OrbitCamera,
        opts: &SceneOpts<'_>,
    ) {
        let scene = Scene { camera, rect };
        let side = self.side();
        draw_floor_and_body(&scene, painter, side);
        if opts.show_reference && opts.view == ViewMode::Anatomical {
            draw_arm(&scene, painter, &neutral_skeleton(side), 0.35, true, false);
        }
        if let Some(round) = opts.round {
            self.draw_round(&scene, painter, round);
        }
        let trail: Vec<Pos2> = self
            .lab
            .trail
            .iter()
            .filter_map(|p| scene.project(*p).map(|q| q.pos))
            .collect();
        if trail.len() > 1 {
            painter.add(Shape::line(
                trail,
                Stroke::new(3.0, rgba(theme::ACCENT, 0.5)),
            ));
        }
        let live = self.tracker.has_fresh_data();
        match opts.view {
            ViewMode::Anatomical => {
                let sk = self.shown_skeleton(side);
                if let Some(sk) = sk {
                    scene.drop_shadow(painter, sk.wrist, 0.06);
                    scene.drop_shadow(painter, sk.blade_tip, 0.04);
                    draw_arm(
                        &scene,
                        painter,
                        &sk,
                        if live { 1.0 } else { 0.4 },
                        false,
                        true,
                    );
                    if !sk.forearm_measured && opts.hud {
                        self.canvas_text(
                            painter,
                            rect.center_top() + Vec2::new(0.0, 60.0),
                            "Forearm sensor not reporting: arm drawn straight, wrist unknown",
                            theme::WARN,
                            15.0,
                        );
                    }
                } else if opts.hud {
                    self.canvas_text(
                        painter,
                        rect.center(),
                        "Calibrate the arm to see it in 3D (or run the simulated demo)",
                        theme::MUTED,
                        18.0,
                    );
                }
            }
            ViewMode::Debug => {
                // Raw measured orientations, not the display-smoothed ones.
                let sk = self.tracker.skeleton();
                if let Some(sk) = sk {
                    self.draw_debug(&scene, painter, &sk, rect, opts);
                } else if opts.hud {
                    self.canvas_text(
                        painter,
                        rect.center(),
                        "Calibrate the arm to see the joint debug view",
                        theme::MUTED,
                        18.0,
                    );
                }
            }
        }
        if opts.hud {
            self.draw_lab_hud(painter, rect, side, opts.round);
        }
    }

    /// Small live 3D arm for screens that are still 2D (Level 0 and the
    /// classic 2D levels). Same renderer and pose as everywhere else.
    pub(crate) fn paint_arm_inset(&self, painter: &egui::Painter, rect: Rect) {
        painter.rect_filled(rect, 12.0, Color32::from_black_alpha(120));
        painter.rect_stroke(
            rect,
            12.0,
            Stroke::new(1.0, theme::FAINT),
            egui::StrokeKind::Inside,
        );
        let clipped = painter.with_clip_rect(rect);
        let opts = SceneOpts {
            round: None,
            show_reference: false,
            view: ViewMode::Anatomical,
            config: self.lab.config,
            hud: false,
        };
        let camera = OrbitCamera {
            distance: 2.5,
            ..OrbitCamera::behind()
        };
        self.paint_scene(&clipped, rect, camera, &opts);
        painter.text(
            rect.left_top() + Vec2::new(8.0, 6.0),
            Align2::LEFT_TOP,
            "3D arm",
            FontId::proportional(12.0),
            theme::MUTED,
        );
    }

    /// Joints, segment axes, hinge axis, hand frame, body frame and the
    /// pattern's reference pose, to make coordinate errors visible.
    fn draw_debug(
        &self,
        scene: &Scene,
        painter: &egui::Painter,
        sk: &ArmSkeleton,
        rect: Rect,
        opts: &SceneOpts<'_>,
    ) {
        let side = self.side();
        let label = |at: Vec3, text: &str, colour: Color32| {
            if let Some(q) = scene.project(at) {
                painter.text(
                    q.pos + Vec2::new(6.0, -6.0),
                    Align2::LEFT_BOTTOM,
                    text,
                    FontId::proportional(12.0),
                    colour,
                );
            }
        };
        let arrow = |from: Vec3, dir: Vec3, length: f64, colour: Color32, text: &str| {
            let tip = from + dir.scale(length);
            scene.line(painter, from, tip, 6.0, colour);
            if let Some(q) = scene.project(tip) {
                painter.circle_filled(q.pos, 3.5, colour);
            }
            label(tip, text, colour);
        };
        let bones = |s: &ArmSkeleton, colour: Color32, width: f32| {
            scene.line(painter, s.shoulder, s.elbow, width, colour);
            scene.line(painter, s.elbow, s.wrist, width, colour);
        };

        // Expected pose: where the pattern puts the arm for the nearest path point.
        let config = opts.config;
        let path = Trajectory::from_config(&config);
        let progress = path.nearest_progress(sk.wrist);
        let reference = config
            .reference_state(progress, false)
            .skeleton(config.side);
        let yellow = Color32::from_rgb(0xFF, 0xD5, 0x4F);
        bones(&reference, rgba(yellow, 0.7), 5.0);
        for p in [reference.elbow, reference.wrist] {
            if let Some(q) = scene.project(p) {
                painter.circle_stroke(q.pos, 6.0, Stroke::new(1.5, yellow));
            }
        }
        label(reference.wrist, "reference", yellow);

        // Body frame at the shoulder.
        for (axis, colour, name) in [
            (Vec3::X, Color32::from_rgb(0xE5, 0x73, 0x73), "x right"),
            (Vec3::Y, Color32::from_rgb(0x81, 0xC7, 0x84), "y fwd"),
            (Vec3::Z, Color32::from_rgb(0x64, 0xB5, 0xF6), "z up"),
        ] {
            arrow(sk.shoulder, axis, 0.3, rgba(colour, 0.7), name);
        }

        // Measured segments and joints.
        bones(sk, Color32::WHITE, 8.0);
        for (p, name) in [
            (sk.shoulder, "shoulder"),
            (sk.elbow, "elbow"),
            (sk.wrist, "wrist"),
        ] {
            if let Some(q) = scene.project(p) {
                painter.circle_filled(q.pos, 7.0, Color32::WHITE);
                painter.circle_stroke(q.pos, 7.0, Stroke::new(2.0, Color32::BLACK));
            }
            label(p, name, Color32::WHITE);
        }
        arrow(
            sk.shoulder,
            sk.upper_dir,
            0.85,
            Color32::from_rgb(0x42, 0xA5, 0xF5),
            "upper-arm axis",
        );
        arrow(
            sk.elbow,
            sk.forearm_dir,
            0.7,
            Color32::from_rgb(0x66, 0xBB, 0x6A),
            "forearm axis",
        );
        let flexion = sk.upper_dir.angle_to(sk.forearm_dir).to_degrees();
        if flexion > 3.0
            && let Some(hinge) = sk.upper_dir.cross(sk.forearm_dir).normalized()
        {
            let magenta = Color32::from_rgb(0xE0, 0x40, 0xFB);
            scene.line(
                painter,
                sk.elbow - hinge.scale(0.2),
                sk.elbow + hinge.scale(0.2),
                7.0,
                magenta,
            );
            label(sk.elbow + hinge.scale(0.2), "elbow flexion axis", magenta);
        }
        // Forearm rotation (pronation/supination) is about the forearm axis.
        let orange = Color32::from_rgb(0xFF, 0xA7, 0x26);
        arrow(sk.wrist, sk.forearm_dir, 0.2, orange, "rotation axis");
        // Hand frame at the wrist: along the fingers, toward the thumb, out of the palm.
        arrow(
            sk.wrist,
            sk.thumb_dir,
            0.2,
            Color32::from_rgb(0xC6, 0xFF, 0x00),
            "thumb side",
        );
        arrow(
            sk.wrist,
            sk.palm_normal,
            0.2,
            Color32::from_rgb(0xFF, 0x6E, 0x40),
            "palm normal",
        );

        // Readouts.
        let (elevation, plane) = elevation_and_plane(sk.upper_dir, side);
        let sim = if self.simulated() {
            "simulated orientations"
        } else {
            "measured orientations"
        };
        let rotation = match self.tracker.forearm_rotation {
            Some(Ok(r)) if self.tracker.has_fresh_data() => {
                format!("{}° {}", signed_deg(r.angle_deg), r.direction.label())
            }
            Some(Err(e)) => e.label().to_string(),
            _ => "not set".to_string(),
        };
        let upper_error = sk.upper_dir.angle_to(reference.upper_dir).to_degrees();
        let wrist_error = (sk.wrist - reference.wrist).length();
        let mut lines = vec![
            format!("DEBUG: {sim} -> forward kinematics"),
            format!(
                "Shoulder orientation: elevation {elevation:.0}°, plane {plane:.0}° (axial rotation: not measured)"
            ),
            format!("Elbow flexion {flexion:.0}°"),
            format!("Forearm pronation/supination {rotation}"),
            format!(
                "Palm normal ({:+.2}, {:+.2}, {:+.2}), facing {} (inferred)",
                sk.palm_normal.x,
                sk.palm_normal.y,
                sk.palm_normal.z,
                limb3d_palm(sk)
            ),
            format!(
                "Wrist endpoint ({:+.2}, {:+.2}, {:+.2}) arm lengths",
                sk.wrist.x, sk.wrist.y, sk.wrist.z
            ),
            format!(
                "vs reference at {:.0}% of the pattern: wrist {wrist_error:.2}, upper arm {upper_error:.0}° off",
                progress * 100.0
            ),
        ];
        if self.simulated()
            && let Some(t) = self.sim.truth()
        {
            // Ground truth of the simulator, shown apart from what the sensors report.
            lines.push(format!(
                "Simulator truth (not a measurement): shoulder quat ({:+.2}, {:+.2}, {:+.2}, {:+.2}), elbow {:.0}°, forearm {}°",
                t.humerus.w,
                t.humerus.x,
                t.humerus.y,
                t.humerus.z,
                t.elbow_deg,
                signed_deg(t.pronation_deg)
            ));
        }
        lines.push("Yellow = pattern template (not patient data)".to_string());
        let mut y = rect.bottom() - 36.0 - 18.0 * lines.len() as f32;
        for line in lines {
            painter.text(
                Pos2::new(rect.left() + 16.0, y),
                Align2::LEFT_TOP,
                line,
                FontId::proportional(13.0),
                theme::TEXT,
            );
            y += 18.0;
        }
    }

    fn canvas_text(
        &self,
        painter: &egui::Painter,
        at: Pos2,
        text: &str,
        color: Color32,
        size: f32,
    ) {
        painter.text(
            at + Vec2::new(1.5, 1.5),
            Align2::CENTER_CENTER,
            text,
            FontId::proportional(size),
            Color32::from_black_alpha(160),
        );
        painter.text(
            at,
            Align2::CENTER_CENTER,
            text,
            FontId::proportional(size),
            color,
        );
    }

    fn draw_round(&self, scene: &Scene, painter: &egui::Painter, round: &Round3d) {
        let path = &round.trajectory;
        if round.config.assistance.shows_guide() {
            let pts: Vec<Pos2> = (0..=40)
                .filter_map(|i| {
                    scene
                        .project(path.point(f64::from(i) / 40.0))
                        .map(|q| q.pos)
                })
                .collect();
            painter.add(Shape::dashed_line(
                &pts,
                Stroke::new(2.5, Color32::from_white_alpha(120)),
                10.0,
                8.0,
            ));
            for t in [0.3, 0.55, 0.8] {
                let a = path.point(t);
                let b = a + path.tangent(t).scale(0.07);
                scene.line(painter, a, b, 9.0, Color32::from_white_alpha(170));
            }
        }
        let active = |phase: Phase| round.phase == phase;
        let start_colour = if matches!(round.phase, Phase::GoToStart | Phase::Return) {
            theme::ACCENT
        } else {
            Color32::from_white_alpha(110)
        };
        let end_colour = if active(Phase::Swipe) {
            Color32::from_rgb(0xFF, 0xD5, 0x4F)
        } else {
            Color32::from_white_alpha(110)
        };
        for (p, colour, label) in [
            (path.start, start_colour, "Start"),
            (path.end, end_colour, "Target"),
        ] {
            scene.drop_shadow(painter, p, 0.1);
            if let Some(q) = scene.project(p) {
                painter.circle_stroke(
                    q.pos,
                    crate::pnf3d::ZONE_ENTER as f32 * q.scale,
                    Stroke::new(3.0, colour),
                );
                painter.text(
                    q.pos + Vec2::new(0.0, -crate::pnf3d::ZONE_ENTER as f32 * q.scale - 12.0),
                    Align2::CENTER_CENTER,
                    label,
                    FontId::proportional(14.0),
                    colour,
                );
            }
        }
        for fruit in round.fruits.iter().filter(|f| !f.sliced) {
            scene.drop_shadow(painter, fruit.pos, fruit.radius * 0.8);
            if let Some(q) = scene.project(fruit.pos) {
                let r = fruit.radius as f32 * q.scale;
                let colour = fruit_colour(fruit.kind);
                painter.circle_filled(q.pos, r, rgba(colour, scene.fade(q.depth)));
                painter.circle_filled(
                    q.pos - Vec2::splat(r * 0.3),
                    r * 0.28,
                    Color32::from_white_alpha(90),
                );
                painter.circle_stroke(q.pos, r, Stroke::new(1.5, Color32::from_black_alpha(110)));
            }
        }
    }

    fn draw_lab_hud(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        side: Side,
        round: Option<&Round3d>,
    ) {
        let pose = self.tracker.pose;
        let mut y = rect.top() + 16.0;
        let mut put = |text: String, color: Color32| {
            painter.text(
                Pos2::new(rect.left() + 16.0, y),
                Align2::LEFT_TOP,
                text,
                FontId::proportional(15.0),
                color,
            );
            y += 21.0;
        };
        if let Some(p) = pose {
            put(
                format!("Shoulder elevation {:.0}°", p.elevation_deg),
                theme::TEXT,
            );
            if let Some(pl) = p.plane_deg {
                put(format!("Plane {pl:.0}° (0 side, 90 front)"), theme::MUTED);
            }
            if let Some(e) = p.elbow_flexion_deg {
                put(format!("Elbow flexion {e:.0}°"), theme::TEXT);
            }
        }
        if let Some(sk) = self.shown_skeleton(side)
            && sk.forearm_measured
            && self.tracker.has_rotation_calibration()
        {
            put(
                format!(
                    "Palm facing {} (inferred)",
                    limb3d::palm_facing(sk.palm_normal)
                ),
                theme::WARN,
            );
        }
        // Forearm rotation block, bottom right
        let centre = rect.right_bottom() + Vec2::new(-130.0, -96.0);
        let est = self.tracker.forearm_rotation;
        rotation_dial(
            painter,
            centre,
            46.0,
            est.and_then(Result::ok)
                .filter(|_| self.tracker.has_fresh_data()),
            side,
        );
        let label = match est {
            Some(Ok(r)) if self.tracker.has_fresh_data() => {
                format!(
                    "{}° {} (estimated)",
                    signed_deg(r.angle_deg),
                    r.direction.label()
                )
            }
            Some(Err(e)) => e.label().to_string(),
            _ => "Not set".to_string(),
        };
        painter.text(
            centre + Vec2::new(0.0, 62.0),
            Align2::CENTER_CENTER,
            "Forearm Rotation",
            FontId::proportional(13.0),
            theme::MUTED,
        );
        painter.text(
            centre + Vec2::new(0.0, 80.0),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(15.0),
            theme::TEXT,
        );
        if let Some(round) = round
            && let Some(rule) = &round.config.rotation
        {
            painter.text(
                centre + Vec2::new(0.0, 98.0),
                Align2::CENTER_CENTER,
                format!(
                    "Goal: {} {:.0}-{:.0}°",
                    rule.expected.label().to_lowercase(),
                    rule.min_deg,
                    rule.max_deg
                ),
                FontId::proportional(12.0),
                theme::FAINT,
            );
        }
    }
}

fn fruit_colour(kind: crate::game::pnf::FruitKind) -> Color32 {
    use crate::game::pnf::FruitKind::*;
    match kind {
        Apple => Color32::from_rgb(0xE5, 0x39, 0x35),
        Orange => Color32::from_rgb(0xFB, 0x8C, 0x00),
        Watermelon => Color32::from_rgb(0x43, 0xA0, 0x47),
        Lemon => Color32::from_rgb(0xFD, 0xD8, 0x35),
        Plum => Color32::from_rgb(0x8E, 0x24, 0xAA),
    }
}

fn rotation_rule_controls(ui: &mut egui::Ui, rule: &mut RotationRule) {
    ui.horizontal(|ui| {
        ui.label("Expected:");
        ui.selectable_value(
            &mut rule.expected,
            RotationDirection::Supination,
            "Supination",
        );
        ui.selectable_value(
            &mut rule.expected,
            RotationDirection::Pronation,
            "Pronation",
        );
    });
    ui.add(egui::Slider::new(&mut rule.min_deg, 1.0..=170.0).text("Smallest rotation °"));
    ui.add(egui::Slider::new(&mut rule.max_deg, 1.0..=180.0).text("Largest rotation °"));
    ui.label(ui::small(
        "Limits are set by the physiotherapist for this patient. There are no universal thresholds.",
    ));
}
