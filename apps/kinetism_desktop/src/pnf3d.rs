//! 3D PNF diagonals: therapist-configurable patterns and the scoring state
//! machine for the 3D fruit game.
//!
//! The reference patterns below are **draft starting points for physiotherapist
//! review**, not clinical prescriptions. Each is a pair of joint-space poses
//! (shoulder plane, elevation and axial rotation, elbow, forearm rotation, see
//! [`crate::armpose`]); the movement between them is a joint-space
//! interpolation (quaternion slerp for the shoulder), and the wrist path is
//! the forward kinematics of that motion. Every value that matters clinically
//! (range, tolerance, expected forearm rotation and its limits, assistance,
//! difficulty) is a setting. Not every patient can complete a textbook
//! pattern, so the individual reach is a setting too.
//!
//! Spatial movement and forearm rotation are scored as two separate
//! components. Neither is an MMT grade, and nothing here is saved.

use crate::arm3d::{ArmSkeleton, segment_point_distance};
use crate::armpose::{ArmState, JointLimits, JointPose, reversed_keys, track};
use crate::clinical::Side;
use crate::forearm::RotationDirection;
use crate::game::pnf::FruitKind;
use crate::math::Vec3;

pub const ZONE_ENTER: f64 = 0.14;
/// Seconds the wrist must stay in the start zone before a rep begins.
pub const START_DWELL: f64 = 0.25;
pub const SWIPE_TIMEOUT: f64 = 30.0;
/// Share of swipe frames that may lack a rotation reading before the rep is not scored.
pub const MAX_MISSING_ROTATION: f64 = 0.25;
const TRAJECTORY_SAMPLES: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnfPattern {
    D1Flexion,
    D1Extension,
    D2Flexion,
    D2Extension,
}

impl PnfPattern {
    pub const ALL: [PnfPattern; 4] = [
        PnfPattern::D1Flexion,
        PnfPattern::D1Extension,
        PnfPattern::D2Flexion,
        PnfPattern::D2Extension,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PnfPattern::D1Flexion => "D1 flexion",
            PnfPattern::D1Extension => "D1 extension",
            PnfPattern::D2Flexion => "D2 flexion",
            PnfPattern::D2Extension => "D2 extension",
        }
    }

    pub fn is_flexion(self) -> bool {
        matches!(self, PnfPattern::D1Flexion | PnfPattern::D2Flexion)
    }

    /// Forearm rotation usually paired with the pattern (flexion with
    /// supination, extension with pronation). A starting suggestion only.
    pub fn suggested_rotation(self) -> RotationDirection {
        if self.is_flexion() {
            RotationDirection::Supination
        } else {
            RotationDirection::Pronation
        }
    }

    /// Draft shoulder poses `(start, end)` in the direction of this pattern,
    /// for the right arm (the left arm uses the same numbers, mirrored).
    ///
    /// Shoulder components, as in PNF:
    /// - D1 flexion: flexion, adduction, external rotation
    /// - D1 extension: extension, abduction, internal rotation
    /// - D2 flexion: flexion, abduction, external rotation
    /// - D2 extension: extension, adduction, internal rotation
    ///
    /// "Plane" is the plane of elevation: 0 = abduction side, 90 = forward,
    /// larger = across the body. The elbow and forearm are separate
    /// trajectories on the same time parameter: see [`Self::elbow_keys`] and
    /// [`Self::pronation_keys`]. These poses carry no elbow or forearm angle.
    pub fn reference_poses(self) -> (JointPose, JointPose) {
        let pose = |plane_deg, elevation_deg, axial_deg| JointPose {
            plane_deg,
            elevation_deg,
            axial_deg,
            elbow_deg: 0.0,
            pronation_deg: 0.0,
        };
        let (lengthened, shortened) = match self {
            // Arm down and out beside the same hip, internally rotated; ends
            // across the face toward the opposite shoulder, externally rotated.
            PnfPattern::D1Flexion | PnfPattern::D1Extension => {
                (pose(10.0, 30.0, -40.0), pose(125.0, 120.0, 40.0))
            }
            // Arm down and across toward the opposite hip, internally rotated;
            // ends up and out above the same shoulder, externally rotated.
            PnfPattern::D2Flexion | PnfPattern::D2Extension => {
                (pose(150.0, 35.0, -40.0), pose(30.0, 135.0, 40.0))
            }
        };
        if self.is_flexion() {
            (lengthened, shortened)
        } else {
            (shortened, lengthened)
        }
    }

    /// Example elbow-flexion trajectory `(time 0..1, degrees)` that goes with
    /// the shoulder movement. **Unvalidated simulation example for
    /// physiotherapist review.**
    /// - D1 flexion: the hand comes up and across toward the opposite shoulder
    ///   and face, so the elbow flexes from nearly straight to a moderate bend and
    ///   eases slightly at the end (a deeper bend with this much external
    ///   rotation folds the forearm back to the same side).
    /// - D2 flexion ("drawing a sword"): the elbow starts bent near the opposite
    ///   hip, bends a little further as the hand clears the body, then
    ///   straightens as the arm reaches up and out.
    /// - Extensions play the same trajectory backwards in time.
    pub fn elbow_keys(self) -> [(f64, f64); 3] {
        let flexion = match self {
            PnfPattern::D1Flexion | PnfPattern::D1Extension => {
                [(0.0, 10.0), (0.5, 50.0), (1.0, 40.0)]
            }
            PnfPattern::D2Flexion | PnfPattern::D2Extension => {
                [(0.0, 45.0), (0.5, 85.0), (1.0, 5.0)]
            }
        };
        if self.is_flexion() {
            flexion
        } else {
            reversed_keys(flexion)
        }
    }

    /// Example forearm rotation trajectory `(time 0..1, degrees, pronation
    /// positive)`. Flexion supinates, extension pronates. **Unvalidated
    /// simulation example for physiotherapist review.**
    pub fn pronation_keys(self) -> [(f64, f64); 3] {
        let flexion = match self {
            PnfPattern::D1Flexion | PnfPattern::D1Extension => {
                [(0.0, 25.0), (0.5, -5.0), (1.0, -45.0)]
            }
            PnfPattern::D2Flexion | PnfPattern::D2Extension => {
                [(0.0, 20.0), (0.5, -20.0), (1.0, -50.0)]
            }
        };
        if self.is_flexion() {
            flexion
        } else {
            reversed_keys(flexion)
        }
    }
}

/// How the elbow behaves during a pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElbowMode {
    /// Held at `PnfConfig::elbow_deg` (0 = a straight-arm pattern).
    Straight,
    /// The pattern's example elbow trajectory, coordinated with the shoulder.
    Coordinated,
}

impl ElbowMode {
    pub const ALL: [ElbowMode; 2] = [ElbowMode::Coordinated, ElbowMode::Straight];

    pub fn label(self) -> &'static str {
        match self {
            ElbowMode::Straight => "Straight / held",
            ElbowMode::Coordinated => "Coordinated example",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Assistance {
    Off,
    /// Show the guide path and direction arrows.
    Guide,
    /// Guide plus wider fruit and tolerance.
    Assisted,
}

impl Assistance {
    pub const ALL: [Assistance; 3] = [Assistance::Off, Assistance::Guide, Assistance::Assisted];

    pub fn label(self) -> &'static str {
        match self {
            Assistance::Off => "None",
            Assistance::Guide => "Guide path",
            Assistance::Assisted => "Guide + wider targets",
        }
    }

    pub fn shows_guide(self) -> bool {
        self != Assistance::Off
    }

    fn widening(self) -> f64 {
        if self == Assistance::Assisted {
            1.5
        } else {
            1.0
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Difficulty {
    Easy,
    Normal,
    Hard,
}

impl Difficulty {
    pub const ALL: [Difficulty; 3] = [Difficulty::Easy, Difficulty::Normal, Difficulty::Hard];

    pub fn label(self) -> &'static str {
        match self {
            Difficulty::Easy => "Easy",
            Difficulty::Normal => "Normal",
            Difficulty::Hard => "Hard",
        }
    }

    fn fruit_radius(self) -> f64 {
        match self {
            Difficulty::Easy => 0.15,
            Difficulty::Normal => 0.11,
            Difficulty::Hard => 0.08,
        }
    }

    fn fruit_count(self) -> usize {
        match self {
            Difficulty::Easy => 3,
            Difficulty::Normal => 4,
            Difficulty::Hard => 5,
        }
    }
}

/// Optional forearm-rotation requirement. The limits are the therapist's, not universal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationRule {
    /// Pronation or supination (never `Neutral`).
    pub expected: RotationDirection,
    /// Smallest rotation toward `expected` that counts, in degrees.
    pub min_deg: f64,
    /// Largest permitted rotation toward `expected`, in degrees.
    pub max_deg: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PnfConfig {
    pub pattern: PnfPattern,
    pub side: Side,
    /// Shift of the start pose: `[plane of elevation, elevation]` in degrees.
    pub start_shift_deg: [f64; 2],
    /// Elbow behaviour: held, or the pattern's coordinated example trajectory.
    pub elbow_mode: ElbowMode,
    /// Elbow flexion held in `ElbowMode::Straight` (0 = straight arm).
    pub elbow_deg: f64,
    /// Share of the template path the patient can reach (0.4-1.0).
    pub reach: f64,
    /// How far the wrist may stray from the path, arm lengths.
    pub tolerance: f64,
    /// `None`: forearm rotation is not assessed.
    pub rotation: Option<RotationRule>,
    pub assistance: Assistance,
    pub difficulty: Difficulty,
    pub reps: u32,
}

impl PnfConfig {
    pub fn new(pattern: PnfPattern, side: Side) -> Self {
        Self {
            pattern,
            side,
            start_shift_deg: [0.0, 0.0],
            elbow_mode: ElbowMode::Coordinated,
            elbow_deg: 0.0,
            reach: 0.9,
            tolerance: 0.14,
            rotation: None,
            assistance: Assistance::Guide,
            difficulty: Difficulty::Normal,
            reps: 5,
        }
    }

    /// Switch on rotation assessment with the pattern's usual direction and
    /// wide placeholder limits for the therapist to adjust.
    pub fn with_suggested_rotation(mut self) -> Self {
        self.rotation = Some(RotationRule {
            expected: self.pattern.suggested_rotation(),
            min_deg: 20.0,
            max_deg: 120.0,
        });
        self
    }

    /// Clamp every setting into a usable range.
    pub fn sanitized(mut self) -> Self {
        let finite = |v: f64, fallback: f64| if v.is_finite() { v } else { fallback };
        self.reach = finite(self.reach, 0.9).clamp(0.4, 1.0);
        self.tolerance = finite(self.tolerance, 0.14).clamp(0.05, 0.4);
        self.reps = self.reps.clamp(1, 20);
        self.start_shift_deg = [
            finite(self.start_shift_deg[0], 0.0).clamp(-30.0, 30.0),
            finite(self.start_shift_deg[1], 0.0).clamp(-30.0, 30.0),
        ];
        self.elbow_deg = finite(self.elbow_deg, 0.0).clamp(0.0, JointLimits::DEFAULT.elbow.1);
        if let Some(rule) = &mut self.rotation {
            if rule.expected == RotationDirection::Neutral {
                rule.expected = self.pattern.suggested_rotation();
            }
            rule.min_deg = finite(rule.min_deg, 20.0).clamp(1.0, 170.0);
            rule.max_deg = finite(rule.max_deg, 120.0).clamp(rule.min_deg, 180.0);
        }
        self
    }

    /// The arm at `progress` (0 = start, 1 = the end of the whole pattern),
    /// scaled by the individual reach. One shared time parameter drives three
    /// separate joint trajectories: the shoulder (quaternion slerp between the
    /// two shoulder poses), the elbow and the forearm rotation. The wrist is
    /// not placed anywhere: it is wherever forward kinematics puts it.
    /// `with_rotation: false` holds the forearm neutral (a spatial-only movement).
    pub fn reference_state(&self, progress: f64, with_rotation: bool) -> ArmState {
        let cfg = self.sanitized();
        let limits = JointLimits::DEFAULT;
        let (mut start, end) = cfg.pattern.reference_poses();
        start.plane_deg += cfg.start_shift_deg[0];
        start.elevation_deg += cfg.start_shift_deg[1];
        let (start, end) = (limits.clamp(start), limits.clamp(end));
        let u = progress.clamp(0.0, 1.0) * cfg.reach;
        let elbow = match cfg.elbow_mode {
            ElbowMode::Straight => cfg.elbow_deg,
            ElbowMode::Coordinated => track(&cfg.pattern.elbow_keys(), u),
        };
        let pronation = if with_rotation {
            track(&cfg.pattern.pronation_keys(), u)
        } else {
            0.0
        };
        let limited = limits.clamp(JointPose {
            elbow_deg: elbow,
            pronation_deg: pronation,
            ..JointPose::REST
        });
        ArmState {
            humerus: start.humerus(cfg.side).slerp(end.humerus(cfg.side), u),
            elbow_deg: limited.elbow_deg,
            pronation_deg: limited.pronation_deg,
        }
    }

    fn tolerance_effective(&self) -> f64 {
        self.tolerance * self.assistance.widening()
    }

    fn fruit_radius_effective(&self) -> f64 {
        self.difficulty.fruit_radius() * self.assistance.widening()
    }
}

/// The wrist path of a pattern: forward kinematics of the joint-space
/// reference motion, sampled. The simulator walks the same motion, so the
/// game's path and the simulated wrist are the same positions by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trajectory {
    pub start: Vec3,
    pub end: Vec3,
    points: [Vec3; TRAJECTORY_SAMPLES + 1],
}

impl Trajectory {
    pub fn from_config(cfg: &PnfConfig) -> Self {
        let cfg = cfg.sanitized();
        let mut points = [Vec3::ZERO; TRAJECTORY_SAMPLES + 1];
        for (i, p) in points.iter_mut().enumerate() {
            *p = cfg
                .reference_state(i as f64 / TRAJECTORY_SAMPLES as f64, false)
                .skeleton(cfg.side)
                .wrist;
        }
        Self {
            start: points[0],
            end: points[TRAJECTORY_SAMPLES],
            points,
        }
    }

    pub fn point(&self, t: f64) -> Vec3 {
        let x = t.clamp(0.0, 1.0) * TRAJECTORY_SAMPLES as f64;
        let i = (x.floor() as usize).min(TRAJECTORY_SAMPLES - 1);
        let f = x - i as f64;
        self.points[i] + (self.points[i + 1] - self.points[i]).scale(f)
    }

    pub fn tangent(&self, t: f64) -> Vec3 {
        let x = t.clamp(0.0, 1.0) * TRAJECTORY_SAMPLES as f64;
        let i = (x.floor() as usize).min(TRAJECTORY_SAMPLES - 1);
        (self.points[i + 1] - self.points[i])
            .normalized()
            .unwrap_or(Vec3::Y)
    }

    /// Distance from `p` to the path.
    pub fn distance(&self, p: Vec3) -> f64 {
        self.points
            .windows(2)
            .map(|w| segment_point_distance(w[0], w[1], p))
            .fold(f64::INFINITY, f64::min)
    }

    /// Progress (0..=1) of the path point nearest to `p`.
    pub fn nearest_progress(&self, p: Vec3) -> f64 {
        let mut best = (f64::INFINITY, 0.0);
        for (i, w) in self.points.windows(2).enumerate() {
            let ab = w[1] - w[0];
            let len2 = ab.dot(ab).max(1e-12);
            let f = ((p - w[0]).dot(ab) / len2).clamp(0.0, 1.0);
            let d = (p - (w[0] + ab.scale(f))).length();
            if d < best.0 {
                best = (d, (i as f64 + f) / TRAJECTORY_SAMPLES as f64);
            }
        }
        best.1
    }

    /// Farthest distance of any path point from the shoulder.
    #[cfg(test)]
    pub fn max_radius(&self) -> f64 {
        self.points.iter().map(|p| p.length()).fold(0.0, f64::max)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    GoToStart,
    Swipe,
    Return,
    Done,
}

#[derive(Clone, Debug)]
pub struct Fruit3d {
    pub kind: FruitKind,
    pub pos: Vec3,
    pub radius: f64,
    pub sliced: bool,
}

/// One tracked sample. `rotation_deg` is the estimated forearm rotation
/// (pronation positive) or `None` when it is unavailable or unreliable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub wrist: Vec3,
    pub blade_tip: Vec3,
    pub rotation_deg: Option<f64>,
}

impl Frame {
    /// `None` when the forearm is not measured: without it the wrist position
    /// would only be a guess, so the game treats it as lost tracking.
    pub fn from_skeleton(skeleton: &ArmSkeleton, rotation_deg: Option<f64>) -> Option<Self> {
        skeleton.forearm_measured.then_some(Self {
            wrist: skeleton.wrist,
            blade_tip: skeleton.blade_tip,
            rotation_deg,
        })
    }

    fn is_finite(&self) -> bool {
        [self.wrist, self.blade_tip]
            .iter()
            .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationOutcome {
    /// Rotation is not assessed in this round.
    Disabled,
    Reached,
    /// Less rotation than the therapist asked for, or the wrong direction.
    NotReached,
    /// More than the permitted range.
    Exceeded,
    /// Too little reliable data to say.
    Unavailable,
}

/// Judge a swipe's rotation against the therapist's rule.
/// `peak_toward_deg` is the largest rotation toward the expected direction
/// (negative when the forearm only turned the other way).
pub fn evaluate_rotation(
    rule: Option<&RotationRule>,
    peak_toward_deg: f64,
    missing_fraction: f64,
) -> RotationOutcome {
    let Some(rule) = rule else {
        return RotationOutcome::Disabled;
    };
    if missing_fraction > MAX_MISSING_ROTATION || !peak_toward_deg.is_finite() {
        RotationOutcome::Unavailable
    } else if peak_toward_deg < rule.min_deg {
        RotationOutcome::NotReached
    } else if peak_toward_deg > rule.max_deg {
        RotationOutcome::Exceeded
    } else {
        RotationOutcome::Reached
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepResult {
    // Spatial component
    pub reached_top: bool,
    pub fruits_sliced: u32,
    pub fruits_total: u32,
    /// Mean distance of the wrist from the path during the swipe (arm lengths).
    pub mean_deviation: f64,
    /// Share of swipe samples within the trajectory tolerance.
    pub within_tolerance: f64,
    pub spatial_score: f64,
    // Rotation component, recorded separately
    pub rotation: RotationOutcome,
    pub peak_rotation_deg: Option<f64>,
}

impl RepResult {
    /// Supportive, separate feedback for the movement and the rotation.
    pub fn feedback(&self) -> Vec<&'static str> {
        let mut lines = Vec::new();
        lines.push(if self.reached_top {
            "Target reached"
        } else {
            "Target not reached yet. Keep following the path"
        });
        match self.rotation {
            RotationOutcome::Disabled | RotationOutcome::Reached => {}
            RotationOutcome::NotReached => lines.push("Forearm rotation target not reached"),
            RotationOutcome::Exceeded => lines.push("Forearm rotation went past the set range"),
            RotationOutcome::Unavailable => lines.push("Forearm rotation unavailable"),
        }
        lines
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PnfEvent {
    RepStarted(u32),
    Sliced {
        pos: Vec3,
        kind: FruitKind,
    },
    ReachedTop,
    RepScored(RepResult),
    /// The rep had a tracking gap or unusable rotation data and was not counted.
    RepDiscarded,
    TrackingLost,
    Finished,
}

#[derive(Clone, Debug)]
pub struct Round3d {
    pub config: PnfConfig,
    pub trajectory: Trajectory,
    pub phase: Phase,
    pub fruits: Vec<Fruit3d>,
    pub rep: u32,
    pub results: Vec<RepResult>,
    pub discarded: u32,
    pub points: i64,
    phase_time: f64,
    start_dwell: f64,
    prev: Option<Frame>,
    tracking_ok: bool,
    // Per-rep accumulators
    reached_top: bool,
    rep_invalid: bool,
    swipe_frames: u32,
    missing_rotation: u32,
    peak_toward: f64,
    deviation_sum: f64,
    within_tolerance: u32,
}

impl Round3d {
    pub fn new(config: PnfConfig) -> Self {
        let config = config.sanitized();
        let trajectory = Trajectory::from_config(&config);
        let mut round = Self {
            config,
            trajectory,
            phase: Phase::GoToStart,
            fruits: Vec::new(),
            rep: 0,
            results: Vec::new(),
            discarded: 0,
            points: 0,
            phase_time: 0.0,
            start_dwell: 0.0,
            prev: None,
            tracking_ok: true,
            reached_top: false,
            rep_invalid: false,
            swipe_frames: 0,
            missing_rotation: 0,
            peak_toward: f64::NEG_INFINITY,
            deviation_sum: 0.0,
            within_tolerance: 0,
        };
        round.spawn_fruit();
        round
    }

    fn spawn_fruit(&mut self) {
        let n = self.config.difficulty.fruit_count();
        let radius = self.config.fruit_radius_effective();
        self.fruits = (0..n)
            .map(|i| Fruit3d {
                kind: FruitKind::ALL[i % FruitKind::ALL.len()],
                pos: self
                    .trajectory
                    .point(0.3 + 0.6 * (i as f64 + 0.5) / n as f64),
                radius,
                sliced: false,
            })
            .collect();
    }

    fn begin_rep(&mut self) {
        self.rep += 1;
        self.phase = Phase::Swipe;
        self.phase_time = 0.0;
        self.reached_top = false;
        self.rep_invalid = false;
        self.swipe_frames = 0;
        self.missing_rotation = 0;
        self.peak_toward = f64::NEG_INFINITY;
        self.deviation_sum = 0.0;
        self.within_tolerance = 0;
        self.spawn_fruit();
    }

    pub fn scored_reps(&self) -> u32 {
        self.results.len() as u32
    }

    /// Mean spatial score over the requested reps (unfinished reps count as 0).
    pub fn spatial_fraction(&self) -> f64 {
        let total = f64::from(self.config.reps);
        // `+ 0.0`: the sum of an empty list is -0.0, which would print as "-0%".
        (self.results.iter().map(|r| r.spatial_score).sum::<f64>() / total + 0.0).clamp(0.0, 1.0)
    }

    /// Share of rotation-assessed reps whose rotation was reached. `None`
    /// when rotation is not assessed or no rep was assessed.
    pub fn rotation_rate(&self) -> Option<f64> {
        self.config.rotation?;
        let assessed: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.rotation != RotationOutcome::Disabled)
            .collect();
        if assessed.is_empty() {
            return None;
        }
        let reached = assessed
            .iter()
            .filter(|r| r.rotation == RotationOutcome::Reached)
            .count();
        Some(reached as f64 / assessed.len() as f64)
    }

    /// Tracking was lost: the next sample starts a new stroke, and a rep in
    /// progress can no longer count.
    pub fn interrupt(&mut self) {
        self.prev = None;
        self.start_dwell = 0.0;
        if matches!(self.phase, Phase::Swipe | Phase::Return) {
            self.rep_invalid = true;
        }
    }

    /// Advance with one sample. `None` means tracking is unavailable (sensor
    /// gap, invalid data or calibration): nothing is scored.
    pub fn update(&mut self, frame: Option<Frame>, dt: f64) -> Vec<PnfEvent> {
        let mut events = Vec::new();
        if self.phase == Phase::Done {
            return events;
        }
        let Some(frame) = frame.filter(Frame::is_finite) else {
            if self.tracking_ok {
                events.push(PnfEvent::TrackingLost);
            }
            self.tracking_ok = false;
            self.interrupt();
            return events;
        };
        self.tracking_ok = true;
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        self.phase_time += dt;
        let contiguous = self.prev.is_some();
        self.prev = Some(frame);
        let wrist = frame.wrist;
        match self.phase {
            Phase::GoToStart => {
                if (wrist - self.trajectory.start).length() <= ZONE_ENTER {
                    self.start_dwell += dt;
                    if self.start_dwell >= START_DWELL {
                        self.start_dwell = 0.0;
                        self.begin_rep();
                        events.push(PnfEvent::RepStarted(self.rep));
                    }
                } else {
                    self.start_dwell = 0.0;
                }
            }
            Phase::Swipe => {
                self.swipe(frame, contiguous, &mut events);
                if self.phase_time > SWIPE_TIMEOUT && self.phase == Phase::Swipe {
                    self.finish_rep(&mut events);
                }
            }
            Phase::Return => {
                if (wrist - self.trajectory.start).length() <= ZONE_ENTER {
                    self.finish_rep(&mut events);
                }
            }
            Phase::Done => {}
        }
        events
    }

    fn swipe(&mut self, frame: Frame, contiguous: bool, events: &mut Vec<PnfEvent>) {
        let tolerance = self.config.tolerance_effective();
        let deviation = self.trajectory.distance(frame.wrist);
        self.swipe_frames += 1;
        self.deviation_sum += deviation;
        if deviation <= tolerance {
            self.within_tolerance += 1;
        }
        if let Some(rule) = &self.config.rotation {
            match frame.rotation_deg {
                Some(angle) => {
                    let toward = if rule.expected == RotationDirection::Pronation {
                        angle
                    } else {
                        -angle
                    };
                    self.peak_toward = self.peak_toward.max(toward);
                }
                None => self.missing_rotation += 1,
            }
        }
        // Only a stroke that stays on the path slices fruit, and never across a tracking gap.
        if contiguous && deviation <= tolerance {
            for fruit in self.fruits.iter_mut().filter(|f| !f.sliced) {
                if segment_point_distance(frame.wrist, frame.blade_tip, fruit.pos) <= fruit.radius {
                    fruit.sliced = true;
                    self.points += 100;
                    events.push(PnfEvent::Sliced {
                        pos: fruit.pos,
                        kind: fruit.kind,
                    });
                }
            }
        }
        if !self.reached_top && (frame.wrist - self.trajectory.end).length() <= ZONE_ENTER {
            self.reached_top = true;
            self.phase = Phase::Return;
            self.phase_time = 0.0;
            events.push(PnfEvent::ReachedTop);
        }
    }

    fn finish_rep(&mut self, events: &mut Vec<PnfEvent>) {
        let missing_fraction = if self.swipe_frames == 0 {
            1.0
        } else {
            f64::from(self.missing_rotation) / f64::from(self.swipe_frames)
        };
        let rotation = evaluate_rotation(
            self.config.rotation.as_ref(),
            self.peak_toward,
            missing_fraction,
        );
        if self.rep_invalid || rotation == RotationOutcome::Unavailable {
            self.discarded += 1;
            self.rep = self.rep.saturating_sub(1);
            self.phase = Phase::GoToStart;
            self.phase_time = 0.0;
            events.push(PnfEvent::RepDiscarded);
            return;
        }
        let frames = f64::from(self.swipe_frames.max(1));
        let mean_deviation = self.deviation_sum / frames;
        let within_tolerance = f64::from(self.within_tolerance) / frames;
        let sliced = self.fruits.iter().filter(|f| f.sliced).count() as u32;
        let total = self.fruits.len() as u32;
        let fruit_fraction = f64::from(sliced) / f64::from(total.max(1));
        let accuracy =
            (1.0 - mean_deviation / (2.0 * self.config.tolerance_effective())).clamp(0.0, 1.0);
        let spatial_score = if self.reached_top {
            0.4 + 0.4 * fruit_fraction + 0.2 * accuracy
        } else {
            0.4 * fruit_fraction * accuracy
        };
        let result = RepResult {
            reached_top: self.reached_top,
            fruits_sliced: sliced,
            fruits_total: total,
            mean_deviation,
            within_tolerance,
            spatial_score,
            rotation,
            peak_rotation_deg: self.peak_toward.is_finite().then_some(self.peak_toward),
        };
        if result.reached_top {
            self.points += 100;
        }
        events.push(PnfEvent::RepScored(result.clone()));
        self.results.push(result);
        if self.scored_reps() >= self.config.reps {
            self.phase = Phase::Done;
            events.push(PnfEvent::Finished);
        } else {
            self.phase = Phase::GoToStart;
        }
        self.phase_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::armpose::track;

    fn cfg(pattern: PnfPattern, side: Side) -> PnfConfig {
        PnfConfig {
            reps: 2,
            ..PnfConfig::new(pattern, side)
        }
    }

    /// A frame whose wrist is at `p` and whose blade points along `dir`.
    fn frame_at(p: Vec3, dir: Vec3, rotation: Option<f64>) -> Frame {
        Frame {
            wrist: p,
            blade_tip: p + dir.scale(0.42),
            rotation_deg: rotation,
        }
    }

    /// Drive one full rep along the path: start -> end -> start.
    fn drive_rep(round: &mut Round3d, rotation: impl Fn(f64) -> Option<f64>) -> Vec<PnfEvent> {
        let mut events = Vec::new();
        let path = round.trajectory;
        for _ in 0..8 {
            events.extend(round.update(
                Some(frame_at(path.start, path.tangent(0.0), rotation(0.0))),
                0.05,
            ));
        }
        for i in 1..=60 {
            let t = i as f64 / 60.0;
            events.extend(round.update(
                Some(frame_at(path.point(t), path.tangent(t), rotation(t))),
                0.05,
            ));
        }
        for i in 1..=20 {
            let p = path.point(1.0 - i as f64 / 20.0);
            events.extend(round.update(Some(frame_at(p, -path.tangent(0.5), None)), 0.05));
        }
        events
    }

    fn scored(events: &[PnfEvent]) -> Vec<&RepResult> {
        events
            .iter()
            .filter_map(|e| match e {
                PnfEvent::RepScored(r) => Some(r),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_template_path_stays_inside_the_reach_sphere_for_both_arms() {
        for pattern in PnfPattern::ALL {
            for side in [Side::Right, Side::Left] {
                for reach in [0.4, 0.7, 1.0] {
                    let t = Trajectory::from_config(&PnfConfig {
                        reach,
                        ..PnfConfig::new(pattern, side)
                    });
                    for i in 0..=20 {
                        let p = t.point(i as f64 / 20.0);
                        assert!(
                            p.length() <= 1.0 + 1e-9,
                            "{pattern:?} {side:?} reach {reach}: {} at {i}",
                            p.length()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn left_arm_paths_mirror_the_right_and_extension_reverses_flexion() {
        let right = Trajectory::from_config(&PnfConfig::new(PnfPattern::D2Flexion, Side::Right));
        let left = Trajectory::from_config(&PnfConfig::new(PnfPattern::D2Flexion, Side::Left));
        assert!(
            (right.start.x + left.start.x).abs() < 1e-12
                && (right.start.y - left.start.y).abs() < 1e-12
        );
        assert!((right.end.x + left.end.x).abs() < 1e-12);
        let ext = Trajectory::from_config(&PnfConfig {
            reach: 1.0,
            ..PnfConfig::new(PnfPattern::D2Extension, Side::Right)
        });
        let flex = Trajectory::from_config(&PnfConfig {
            reach: 1.0,
            ..PnfConfig::new(PnfPattern::D2Flexion, Side::Right)
        });
        assert!((ext.start - flex.end).length() < 1e-9);
        assert!((ext.end - flex.start).length() < 1e-9);
    }

    #[test]
    fn d2_flexion_rises_up_and_out_and_d1_crosses_the_body() {
        let d2 = Trajectory::from_config(&PnfConfig::new(PnfPattern::D2Flexion, Side::Right));
        assert!(d2.end.z > d2.start.z + 0.8, "D2 rises");
        assert!(
            d2.end.x > 0.3 && d2.start.x < 0.0,
            "D2 goes from the opposite hip out to the same side"
        );
        let d1 = Trajectory::from_config(&PnfConfig::new(PnfPattern::D1Flexion, Side::Right));
        assert!(
            d1.start.x > 0.0 && d1.end.x < 0.0,
            "D1 crosses to the opposite side"
        );
    }

    #[test]
    fn sanitizing_clamps_every_setting_and_repairs_a_neutral_rotation_rule() {
        let c = PnfConfig {
            reach: f64::NAN,
            tolerance: 9.0,
            reps: 0,
            start_shift_deg: [99.0, f64::NAN],
            elbow_deg: 999.0,
            rotation: Some(RotationRule {
                expected: RotationDirection::Neutral,
                min_deg: 400.0,
                max_deg: -5.0,
            }),
            ..PnfConfig::new(PnfPattern::D1Extension, Side::Right)
        }
        .sanitized();
        assert_eq!(c.reach, 0.9);
        assert_eq!(c.tolerance, 0.4);
        assert_eq!(c.reps, 1);
        assert_eq!(c.start_shift_deg, [30.0, 0.0]);
        assert_eq!(c.elbow_deg, 145.0, "elbow limit");
        let rule = c.rotation.unwrap();
        assert_eq!(
            rule.expected,
            RotationDirection::Pronation,
            "extension suggests pronation"
        );
        assert!(rule.min_deg <= rule.max_deg && rule.max_deg <= 180.0);
    }

    fn wrists(pattern: PnfPattern, side: Side, mode: ElbowMode) -> Vec<Vec3> {
        let cfg = PnfConfig {
            reach: 1.0,
            elbow_mode: mode,
            ..PnfConfig::new(pattern, side)
        };
        (0..=24)
            .map(|i| {
                cfg.reference_state(f64::from(i) / 24.0, true)
                    .skeleton(side)
                    .wrist
            })
            .collect()
    }

    #[test]
    fn reference_poses_carry_the_pnf_shoulder_components() {
        // plane: + = toward adduction across the body, - = toward abduction.
        for (pattern, plane, elevation, axial) in [
            (PnfPattern::D1Flexion, 1.0, 1.0, 1.0),
            (PnfPattern::D1Extension, -1.0, -1.0, -1.0),
            (PnfPattern::D2Flexion, -1.0, 1.0, 1.0),
            (PnfPattern::D2Extension, 1.0, -1.0, -1.0),
        ] {
            let (a, b) = pattern.reference_poses();
            assert_eq!(
                (
                    (b.plane_deg - a.plane_deg).signum(),
                    (b.elevation_deg - a.elevation_deg).signum(),
                    (b.axial_deg - a.axial_deg).signum()
                ),
                (plane, elevation, axial),
                "{pattern:?}"
            );
            assert_eq!(
                (a.elbow_deg, b.elbow_deg),
                (0.0, 0.0),
                "straight-arm reference"
            );
        }
    }

    #[test]
    fn flexion_ends_supinated_and_extension_ends_pronated() {
        for pattern in PnfPattern::ALL {
            let keys = pattern.pronation_keys();
            let (first, last) = (keys[0].1, keys[2].1);
            assert!(
                first.signum() != last.signum(),
                "{pattern:?} turns through neutral"
            );
            let ended_pronated = last > 0.0;
            assert_eq!(ended_pronated, !pattern.is_flexion(), "{pattern:?}");
            assert_eq!(
                pattern.suggested_rotation(),
                if ended_pronated {
                    RotationDirection::Pronation
                } else {
                    RotationDirection::Supination
                }
            );
        }
    }

    #[test]
    fn d2_path_runs_from_the_opposite_hip_up_and_out_above_the_same_shoulder() {
        for mode in [ElbowMode::Coordinated, ElbowMode::Straight] {
            for side in [Side::Right, Side::Left] {
                let s = side.sign();
                let w = wrists(PnfPattern::D2Flexion, side, mode);
                let (first, last) = (w[0], w[w.len() - 1]);
                assert!(
                    first.x * s < -0.4 && first.z < -0.25,
                    "{mode:?} {side:?} starts across the body low down: {first:?}"
                );
                assert!(
                    last.x * s > 0.4 && last.z > 0.5,
                    "{mode:?} {side:?} ends up and out on the same side: {last:?}"
                );
                assert!(
                    w.iter().all(|p| p.y > 0.0),
                    "the whole diagonal is in front of the body"
                );
                let mid = w[12];
                assert!(mid.z > first.z && mid.z < last.z, "{mode:?} {mid:?}");
                assert!(
                    mid.x * s > first.x * s && mid.x * s < last.x * s + 0.2,
                    "{mode:?} {mid:?}"
                );
            }
        }
    }

    #[test]
    fn d1_path_runs_from_the_same_hip_across_the_body_to_the_opposite_shoulder() {
        for mode in [ElbowMode::Coordinated, ElbowMode::Straight] {
            for side in [Side::Right, Side::Left] {
                let s = side.sign();
                let w = wrists(PnfPattern::D1Flexion, side, mode);
                let (first, last) = (w[0], w[w.len() - 1]);
                assert!(
                    first.x * s > 0.3 && first.z < -0.6,
                    "{mode:?} {side:?} starts beside the same hip: {first:?}"
                );
                assert!(
                    last.x * s < -0.1 && last.z > 0.3 && last.y > 0.3,
                    "{mode:?} {side:?} ends in front of the face toward the opposite side: {last:?}"
                );
                assert!(
                    w.iter().any(|p| p.x * s < -0.3),
                    "{mode:?} {side:?} crosses the body"
                );
                assert!(w[12].z > first.z, "{mode:?} rises");
            }
        }
    }

    #[test]
    fn extension_patterns_retrace_flexion_in_reverse() {
        for mode in [ElbowMode::Coordinated, ElbowMode::Straight] {
            for (flexion, extension) in [
                (PnfPattern::D1Flexion, PnfPattern::D1Extension),
                (PnfPattern::D2Flexion, PnfPattern::D2Extension),
            ] {
                let f = wrists(flexion, Side::Right, mode);
                let e = wrists(extension, Side::Right, mode);
                for (a, b) in f.iter().zip(e.iter().rev()) {
                    assert!((*a - *b).length() < 1e-9, "{mode:?}");
                }
            }
        }
    }

    #[test]
    fn reference_motion_is_smooth_and_the_wrist_never_leaves_arm_reach() {
        for pattern in PnfPattern::ALL {
            for side in [Side::Right, Side::Left] {
                for mode in [ElbowMode::Coordinated, ElbowMode::Straight] {
                    let cfg = PnfConfig {
                        reach: 1.0,
                        elbow_mode: mode,
                        ..PnfConfig::new(pattern, side)
                    };
                    let mut prev: Option<(crate::armpose::ArmState, ArmSkeleton)> = None;
                    for i in 0..=100 {
                        let st = cfg.reference_state(f64::from(i) / 100.0, true);
                        let sk = st.skeleton(side);
                        assert!(sk.wrist.length() <= 1.0 + 1e-9, "{pattern:?} {side:?}");
                        if mode == ElbowMode::Straight {
                            assert!(st.elbow_deg.abs() < 1e-12, "held at the straight setting");
                        }
                        if let Some((pst, p)) = prev {
                            let who = format!("{pattern:?} {side:?} {mode:?} at {i}");
                            assert!((sk.wrist - p.wrist).length() < 0.06, "wrist jump {who}");
                            assert!(
                                sk.palm_normal.angle_to(p.palm_normal).to_degrees() < 3.5,
                                "palm flip {who}"
                            );
                            assert!(
                                sk.forearm_dir.angle_to(p.forearm_dir).to_degrees() < 4.5,
                                "forearm {who}"
                            );
                            assert!(
                                (st.elbow_deg - pst.elbow_deg).abs() < 2.5,
                                "elbow step {who}"
                            );
                            assert!(
                                (st.pronation_deg - pst.pronation_deg).abs() < 2.5,
                                "forearm step {who}"
                            );
                        }
                        prev = Some((st, sk));
                    }
                }
            }
        }
    }

    #[test]
    fn a_held_elbow_bends_as_a_hinge_and_pulls_the_wrist_in() {
        let side = Side::Right;
        let base = PnfConfig {
            reach: 1.0,
            elbow_mode: ElbowMode::Straight,
            ..PnfConfig::new(PnfPattern::D2Flexion, side)
        };
        let bent = PnfConfig {
            elbow_deg: 60.0,
            ..base
        };
        for i in 0..=10 {
            let t = f64::from(i) / 10.0;
            let st = bent.reference_state(t, false);
            let sk = st.skeleton(side);
            assert!(
                sk.wrist.length() < 1.0 - 0.05,
                "a bent elbow shortens the reach"
            );
            let (u, f) = st.directions();
            assert!((u.angle_to(f).to_degrees() - 60.0).abs() < 1e-6);
            assert!(f.dot(st.hinge_axis()).abs() < 1e-9, "hinge motion only");
        }
        let upper = |c: &PnfConfig| c.reference_state(0.5, false).skeleton(side).upper_dir;
        assert!(
            upper(&bent).angle_to(upper(&base)) < 1e-9,
            "the shoulder motion is unchanged"
        );
    }

    #[test]
    fn reach_and_start_shift_adjust_the_range_in_joint_space() {
        let side = Side::Right;
        let make = |reach, shift| {
            Trajectory::from_config(&PnfConfig {
                reach,
                start_shift_deg: shift,
                ..PnfConfig::new(PnfPattern::D2Flexion, side)
            })
        };
        let full = make(1.0, [0.0, 0.0]);
        let half = make(0.5, [0.0, 0.0]);
        assert!((full.start - half.start).length() < 1e-9);
        assert!((full.end - half.end).length() > 0.5);
        assert!(
            (half.end - full.point(0.5)).length() < 1e-9,
            "half the reach is the halfway pose"
        );
        let shifted = make(1.0, [0.0, 20.0]);
        assert!((shifted.start - full.start).length() > 0.1);
        assert!(shifted.max_radius() <= 1.0 + 1e-9);
    }

    #[test]
    fn nearest_progress_recovers_where_on_the_path_a_point_is() {
        let t = Trajectory::from_config(&PnfConfig::new(PnfPattern::D1Flexion, Side::Right));
        for p in [0.0, 0.25, 0.6, 1.0] {
            assert!((t.nearest_progress(t.point(p)) - p).abs() < 0.02, "{p}");
        }
    }

    #[test]
    fn the_elbow_moves_during_every_coordinated_pattern_and_is_held_when_straight() {
        for pattern in PnfPattern::ALL {
            for side in [Side::Right, Side::Left] {
                let at = |mode, u| {
                    PnfConfig {
                        reach: 1.0,
                        elbow_mode: mode,
                        ..PnfConfig::new(pattern, side)
                    }
                    .reference_state(u, true)
                    .elbow_deg
                };
                let samples: Vec<f64> = (0..=50)
                    .map(|i| at(ElbowMode::Coordinated, f64::from(i) / 50.0))
                    .collect();
                let (lo, hi) = samples
                    .iter()
                    .fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
                assert!(
                    hi - lo >= 30.0,
                    "{pattern:?} {side:?}: elbow only moved {} deg",
                    hi - lo
                );
                assert!(lo >= 0.0 && hi <= 145.0, "within the elbow limits");
                assert!(
                    (0..=50).all(|i| at(ElbowMode::Straight, f64::from(i) / 50.0).abs() < 1e-12),
                    "the straight-elbow variation stays available"
                );
            }
        }
    }

    #[test]
    fn the_elbow_follows_the_documented_trajectory_for_each_pattern() {
        let elbow = |pattern, u| {
            PnfConfig {
                reach: 1.0,
                ..PnfConfig::new(pattern, Side::Right)
            }
            .reference_state(u, true)
            .elbow_deg
        };
        // D1 flexion: nearly straight at the hip, bent mid-way, eases at the end.
        assert!(elbow(PnfPattern::D1Flexion, 0.0) < 15.0);
        assert!(elbow(PnfPattern::D1Flexion, 0.5) > 45.0);
        assert!(elbow(PnfPattern::D1Flexion, 1.0) < elbow(PnfPattern::D1Flexion, 0.5));
        // D2 flexion: bent at the opposite hip, bends further, straightens overhead.
        assert!(elbow(PnfPattern::D2Flexion, 0.0) > 30.0);
        assert!(elbow(PnfPattern::D2Flexion, 0.5) > elbow(PnfPattern::D2Flexion, 0.0));
        assert!(elbow(PnfPattern::D2Flexion, 1.0) < 10.0);
        // Different patterns have different elbow trajectories.
        assert!(
            (elbow(PnfPattern::D1Flexion, 0.0) - elbow(PnfPattern::D2Flexion, 0.0)).abs() > 20.0
        );
        // Extension plays flexion's trajectory backwards.
        for pattern in [PnfPattern::D1Flexion, PnfPattern::D2Flexion] {
            let ext = if pattern == PnfPattern::D1Flexion {
                PnfPattern::D1Extension
            } else {
                PnfPattern::D2Extension
            };
            for i in 0..=10 {
                let u = f64::from(i) / 10.0;
                assert!((elbow(pattern, u) - elbow(ext, 1.0 - u)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn shoulder_elbow_and_forearm_share_one_time_parameter() {
        for pattern in PnfPattern::ALL {
            for side in [Side::Right, Side::Left] {
                for reach in [1.0, 0.7] {
                    let cfg = PnfConfig {
                        reach,
                        ..PnfConfig::new(pattern, side)
                    };
                    let (start, end) = pattern.reference_poses();
                    for i in 0..=10 {
                        let progress = f64::from(i) / 10.0;
                        let u = progress * reach;
                        let st = cfg.reference_state(progress, true);
                        // Each component is its own trajectory evaluated at the same time.
                        let shoulder = start.humerus(side).slerp(end.humerus(side), u);
                        assert!(
                            (st.humerus.conjugate() * shoulder).w.abs() > 1.0 - 1e-12,
                            "{pattern:?} {side:?} reach {reach} at {progress}"
                        );
                        assert!((st.elbow_deg - track(&pattern.elbow_keys(), u)).abs() < 1e-9);
                        assert!(
                            (st.pronation_deg - track(&pattern.pronation_keys(), u)).abs() < 1e-9
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_wrist_is_whatever_forward_kinematics_gives_not_a_placed_point() {
        // Changing only the elbow trajectory moves the wrist; the shoulder is unchanged.
        let side = Side::Right;
        let straight = PnfConfig {
            reach: 1.0,
            elbow_mode: ElbowMode::Straight,
            ..PnfConfig::new(PnfPattern::D2Flexion, side)
        };
        let bent = PnfConfig {
            reach: 1.0,
            ..PnfConfig::new(PnfPattern::D2Flexion, side)
        };
        let a = straight.reference_state(0.3, true).skeleton(side);
        let b = bent.reference_state(0.3, true).skeleton(side);
        assert!((a.upper_dir - b.upper_dir).length() < 1e-9);
        assert!(
            (a.elbow - b.elbow).length() < 1e-9,
            "the elbow joint stays on the upper arm"
        );
        assert!(
            (a.wrist - b.wrist).length() > 0.2,
            "the bend moves the wrist"
        );
        assert!(
            (((b.wrist - b.elbow).length()) - 0.45).abs() < 1e-9,
            "forearm length is constant"
        );
    }

    #[test]
    fn forearm_rotation_changes_during_every_pattern_and_can_be_held_neutral() {
        for pattern in PnfPattern::ALL {
            let cfg = PnfConfig {
                reach: 1.0,
                ..PnfConfig::new(pattern, Side::Right)
            };
            let rot: Vec<f64> = (0..=40)
                .map(|i| cfg.reference_state(f64::from(i) / 40.0, true).pronation_deg)
                .collect();
            let (lo, hi) = rot
                .iter()
                .fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
            assert!(hi - lo >= 60.0, "{pattern:?}: rotation range {}", hi - lo);
            assert!(
                (0..=40).all(|i| cfg
                    .reference_state(f64::from(i) / 40.0, false)
                    .pronation_deg
                    == 0.0),
                "a spatial-only movement holds the forearm neutral"
            );
        }
    }

    #[test]
    fn a_clean_spatial_rep_scores_with_rotation_disabled() {
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        let events = drive_rep(&mut round, |_| None);
        let reps = scored(&events);
        assert_eq!(reps.len(), 1, "{events:?}");
        let rep = reps[0];
        assert!(rep.reached_top);
        assert_eq!(
            rep.fruits_sliced, rep.fruits_total,
            "the blade slices every fruit on the path"
        );
        assert_eq!(rep.rotation, RotationOutcome::Disabled);
        assert!(rep.spatial_score > 0.95, "{}", rep.spatial_score);
        assert_eq!(rep.feedback(), vec!["Target reached"]);
        assert_eq!(round.rotation_rate(), None);
    }

    #[test]
    fn rotation_is_scored_separately_from_the_spatial_movement() {
        let rule = |round: &mut Round3d| {
            round.config.rotation = Some(RotationRule {
                expected: RotationDirection::Supination,
                min_deg: 30.0,
                max_deg: 90.0,
            });
        };
        // Right movement, rotation reached.
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        rule(&mut round);
        let events = drive_rep(&mut round, |t| Some(-60.0 * t));
        let rep = scored(&events)[0].clone();
        assert!(rep.reached_top);
        assert_eq!(rep.rotation, RotationOutcome::Reached);
        let peak = rep.peak_rotation_deg.unwrap();
        assert!(
            (45.0..=60.0).contains(&peak),
            "peak before the target zone: {peak}"
        );
        assert_eq!(round.rotation_rate(), Some(1.0));

        // Right movement, too little rotation: still a successful spatial rep.
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        rule(&mut round);
        let events = drive_rep(&mut round, |t| Some(-10.0 * t));
        let rep = scored(&events)[0].clone();
        assert!(rep.reached_top && rep.spatial_score > 0.95);
        assert_eq!(rep.rotation, RotationOutcome::NotReached);
        assert_eq!(
            rep.feedback(),
            vec!["Target reached", "Forearm rotation target not reached"]
        );

        // Wrong direction counts as not reached, too much as exceeded.
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        rule(&mut round);
        let events = drive_rep(&mut round, |t| Some(60.0 * t));
        assert_eq!(scored(&events)[0].rotation, RotationOutcome::NotReached);
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        rule(&mut round);
        let events = drive_rep(&mut round, |t| Some(-150.0 * t));
        assert_eq!(scored(&events)[0].rotation, RotationOutcome::Exceeded);
    }

    #[test]
    fn rotation_evaluation_boundaries() {
        let rule = RotationRule {
            expected: RotationDirection::Pronation,
            min_deg: 20.0,
            max_deg: 80.0,
        };
        assert_eq!(evaluate_rotation(None, 0.0, 0.0), RotationOutcome::Disabled);
        assert_eq!(
            evaluate_rotation(Some(&rule), 20.0, 0.0),
            RotationOutcome::Reached
        );
        assert_eq!(
            evaluate_rotation(Some(&rule), 80.0, 0.0),
            RotationOutcome::Reached
        );
        assert_eq!(
            evaluate_rotation(Some(&rule), 19.9, 0.0),
            RotationOutcome::NotReached
        );
        assert_eq!(
            evaluate_rotation(Some(&rule), 80.1, 0.0),
            RotationOutcome::Exceeded
        );
        assert_eq!(
            evaluate_rotation(Some(&rule), 50.0, 0.26),
            RotationOutcome::Unavailable
        );
        assert_eq!(
            evaluate_rotation(Some(&rule), f64::NEG_INFINITY, 0.0),
            RotationOutcome::Unavailable
        );
    }

    #[test]
    fn an_incomplete_movement_does_not_reach_the_target_or_finish_the_rep_by_itself() {
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        let path = round.trajectory;
        for _ in 0..8 {
            round.update(Some(frame_at(path.start, path.tangent(0.0), None)), 0.05);
        }
        assert_eq!(round.phase, Phase::Swipe);
        for i in 1..=20 {
            let t = 0.5 * i as f64 / 20.0;
            round.update(Some(frame_at(path.point(t), path.tangent(t), None)), 0.05);
        }
        assert_eq!(round.phase, Phase::Swipe, "halfway is not the target");
        assert!(round.results.is_empty());
        // Give up after the timeout: scored as not reached.
        let events = round.update(
            Some(frame_at(path.point(0.5), path.tangent(0.5), None)),
            SWIPE_TIMEOUT,
        );
        let reps = scored(&events);
        assert_eq!(reps.len(), 1);
        assert!(!reps[0].reached_top);
        assert!(reps[0].spatial_score < 0.4, "{}", reps[0].spatial_score);
        assert_eq!(
            reps[0].feedback()[0],
            "Target not reached yet. Keep following the path"
        );
    }

    #[test]
    fn a_stroke_far_from_the_path_slices_nothing() {
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        let path = round.trajectory;
        for _ in 0..8 {
            round.update(Some(frame_at(path.start, path.tangent(0.0), None)), 0.05);
        }
        let off = Vec3::new(0.0, 0.0, 0.0);
        for i in 1..=60 {
            let t = i as f64 / 60.0;
            // Same path shifted well beyond the tolerance, blade still touches the fruit.
            let p = path.point(t) + Vec3::new(0.0, -0.3, 0.0) + off;
            round.update(Some(frame_at(p, Vec3::new(0.0, 1.0, 0.0), None)), 0.05);
        }
        assert!(round.fruits.iter().all(|f| !f.sliced));
    }

    #[test]
    fn sensor_jitter_cannot_slice_a_fruit_or_start_a_rep_twice() {
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        let path = round.trajectory;
        // Jitter in and out of the start zone faster than the dwell time: no rep.
        for i in 0..40 {
            let p = if i % 2 == 0 {
                path.start
            } else {
                path.start + Vec3::new(0.5, 0.0, 0.0)
            };
            let ev = round.update(Some(frame_at(p, Vec3::Y, None)), 0.1);
            assert!(
                !ev.iter().any(|e| matches!(e, PnfEvent::RepStarted(_))),
                "{ev:?}"
            );
        }
        assert_eq!(round.rep, 0);

        let events = drive_rep(&mut round, |_| None);
        let started = events
            .iter()
            .filter(|e| matches!(e, PnfEvent::RepStarted(_)))
            .count();
        let sliced = events
            .iter()
            .filter(|e| matches!(e, PnfEvent::Sliced { .. }))
            .count();
        let tops = events
            .iter()
            .filter(|e| matches!(e, PnfEvent::ReachedTop))
            .count();
        assert_eq!((started, tops), (1, 1));
        assert_eq!(sliced, round.fruits.len(), "each fruit once: {sliced}");
        assert_eq!(round.scored_reps(), 1, "one scored rep");
        let points_after = round.points;

        // Hovering at the target and shaking the blade changes nothing once the rep is scored.
        for _ in 0..30 {
            round.update(Some(frame_at(path.start, Vec3::Y, None)), 0.01);
        }
        assert_eq!(round.points, points_after);
        assert_eq!(round.scored_reps(), 1);
    }

    #[test]
    fn nothing_is_scored_without_tracking_and_a_rep_with_a_gap_is_discarded() {
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        let path = round.trajectory;
        // No data at all: nothing happens, and the loss is reported once.
        let ev = round.update(None, 0.05);
        assert_eq!(ev, vec![PnfEvent::TrackingLost]);
        assert!(round.update(None, 0.05).is_empty());
        for _ in 0..10 {
            round.update(None, 0.05);
        }
        assert_eq!((round.rep, round.points, round.scored_reps()), (0, 0, 0));
        // Non-finite samples count as lost tracking.
        let bad = Frame {
            wrist: Vec3::new(f64::NAN, 0.0, 0.0),
            blade_tip: Vec3::ZERO,
            rotation_deg: None,
        };
        round.update(Some(bad), 0.05);
        assert_eq!(round.rep, 0);

        // A gap mid-swipe: the rep is discarded and can be repeated.
        for _ in 0..8 {
            round.update(Some(frame_at(path.start, path.tangent(0.0), None)), 0.05);
        }
        assert_eq!(round.rep, 1);
        for i in 1..=30 {
            let t = i as f64 / 60.0;
            round.update(Some(frame_at(path.point(t), path.tangent(t), None)), 0.05);
        }
        round.update(None, 0.05);
        let mut events = Vec::new();
        for i in 31..=60 {
            let t = i as f64 / 60.0;
            events.extend(round.update(Some(frame_at(path.point(t), path.tangent(t), None)), 0.05));
        }
        for i in 1..=20 {
            let p = path.point(1.0 - i as f64 / 20.0);
            events.extend(round.update(Some(frame_at(p, -path.tangent(0.5), None)), 0.05));
        }
        assert!(events.contains(&PnfEvent::RepDiscarded), "{events:?}");
        assert!(
            scored(&events).is_empty(),
            "a rep with a tracking gap never scores"
        );
        assert_eq!((round.discarded, round.rep, round.scored_reps()), (1, 0, 0));
        assert_eq!(round.phase, Phase::GoToStart);
    }

    #[test]
    fn a_gap_never_slices_fruit_across_it() {
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        let path = round.trajectory;
        for _ in 0..8 {
            round.update(Some(frame_at(path.start, path.tangent(0.0), None)), 0.05);
        }
        round.update(None, 0.05);
        let first = &round.fruits[0];
        let ev = round.update(Some(frame_at(first.pos, path.tangent(0.4), None)), 0.05);
        assert!(
            !ev.iter().any(|e| matches!(e, PnfEvent::Sliced { .. })),
            "first sample after a gap"
        );
    }

    #[test]
    fn rotation_enabled_with_unreliable_data_cannot_produce_a_scored_rep() {
        let mut round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        round.config.rotation = Some(RotationRule {
            expected: RotationDirection::Supination,
            min_deg: 20.0,
            max_deg: 120.0,
        });
        let events = drive_rep(&mut round, |_| None);
        assert!(scored(&events).is_empty());
        assert!(events.contains(&PnfEvent::RepDiscarded));
        assert_eq!(
            round.points,
            100 * round.fruits.iter().filter(|f| f.sliced).count() as i64
        );
    }

    #[test]
    fn an_unplayed_round_scores_a_positive_zero() {
        let round = Round3d::new(cfg(PnfPattern::D2Flexion, Side::Right));
        let fraction = round.spatial_fraction();
        assert_eq!(fraction, 0.0);
        assert!(fraction.is_sign_positive(), "would print as -0%");
        assert_eq!(format!("{:.0}", fraction * 100.0), "0");
    }

    #[test]
    fn the_round_finishes_after_the_requested_scored_reps() {
        let mut round = Round3d::new(cfg(PnfPattern::D1Flexion, Side::Left));
        let mut all = Vec::new();
        all.extend(drive_rep(&mut round, |_| None));
        all.extend(drive_rep(&mut round, |_| None));
        assert_eq!(scored(&all).len(), 2);
        assert_eq!(round.phase, Phase::Done);
        assert!(all.contains(&PnfEvent::Finished));
        assert!(
            round
                .update(Some(frame_at(Vec3::ZERO, Vec3::Y, None)), 0.1)
                .is_empty()
        );
        assert!(
            (round.spatial_fraction() - 1.0).abs() < 0.05,
            "{}",
            round.spatial_fraction()
        );
    }

    #[test]
    fn assistance_and_difficulty_change_targets_but_not_the_rules() {
        let easy = Round3d::new(PnfConfig {
            difficulty: Difficulty::Easy,
            assistance: Assistance::Assisted,
            ..PnfConfig::new(PnfPattern::D2Flexion, Side::Right)
        });
        let hard = Round3d::new(PnfConfig {
            difficulty: Difficulty::Hard,
            assistance: Assistance::Off,
            ..PnfConfig::new(PnfPattern::D2Flexion, Side::Right)
        });
        assert!(easy.fruits.len() < hard.fruits.len());
        assert!(easy.fruits[0].radius > hard.fruits[0].radius * 2.0);
        assert!(Assistance::Guide.shows_guide() && !Assistance::Off.shows_guide());
    }
}
