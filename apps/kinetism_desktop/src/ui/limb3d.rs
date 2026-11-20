//! Procedural 3D arm mesh for the 3D view: tapered limb segments, joints, a
//! flattened palm with fingers and a thumb. Built from quads that are
//! projected through the existing camera and drawn with the egui painter
//! (painter's algorithm, back-face culling, one directional light).
//!
//! Visual only. The mesh follows the skeleton from [`crate::arm3d`]; it adds
//! no tracking, scoring or rotation math. The hand orientation is the same
//! inferred, forearm-based orientation the rest of the app uses.
//!
//! Palm vs back of hand: the palm side is light with crease lines, the back is
//! a darker tone with knuckle dots, and the thumb sits on one side only.

use crate::arm3d::ArmSkeleton;
use crate::math::Vec3;
use eframe::egui::Color32;

const SIDES: usize = 10;

pub const SLEEVE: Color32 = Color32::from_rgb(0x2E, 0x8F, 0x83);
pub const SKIN: Color32 = Color32::from_rgb(0xD9, 0xA6, 0x7F);
pub const PALM: Color32 = Color32::from_rgb(0xF6, 0xD6, 0xBA);
pub const BACK_OF_HAND: Color32 = Color32::from_rgb(0x9C, 0x66, 0x47);
const HAND_SIDE: Color32 = Color32::from_rgb(0xC2, 0x8C, 0x68);
const PALM_LINE: Color32 = Color32::from_rgb(0xBB, 0x84, 0x62);
const KNUCKLE: Color32 = Color32::from_rgb(0x7C, 0x4D, 0x34);

/// Palm and finger length along the forearm axis (arm lengths).
pub const PALM_LENGTH: f64 = 0.11;
pub const FINGER_LENGTH: f64 = 0.10;
/// Where the visible hand ends (the blade starts here visually).
pub const HAND_VISUAL_LENGTH: f64 = PALM_LENGTH + FINGER_LENGTH;
const PALM_WIDTH: f64 = 0.125;
const PALM_THICKNESS: f64 = 0.05;

/// One lit quad. A triangle repeats a vertex.
#[derive(Clone, Copy, Debug)]
pub struct Face {
    pub v: [Vec3; 4],
    /// Outward unit normal.
    pub n: Vec3,
    pub color: Color32,
    /// Markings on the hand: drawn just in front of the surface they sit on.
    pub decal: bool,
}

impl Face {
    pub fn center(&self) -> Vec3 {
        (self.v[0] + self.v[1] + self.v[2] + self.v[3]).scale(0.25)
    }

    /// Faces turned away from the eye are not drawn.
    pub fn front_facing(&self, eye: Vec3) -> bool {
        self.n.dot(eye - self.center()) > 0.0
    }
}

/// A quad whose normal is made to point away from `inside`.
fn quad(faces: &mut Vec<Face>, v: [Vec3; 4], inside: Vec3, color: Color32, decal: bool) {
    let normal = (v[1] - v[0]).cross(v[3] - v[0]);
    let normal = if normal.length() < 1e-12 {
        (v[2] - v[1]).cross(v[0] - v[1])
    } else {
        normal
    };
    let Some(mut n) = normal.normalized() else {
        return;
    };
    let centre = (v[0] + v[1] + v[2] + v[3]).scale(0.25);
    if n.dot(centre - inside) < 0.0 {
        n = -n;
    }
    faces.push(Face { v, n, color, decal });
}

/// Two unit vectors perpendicular to `axis` (and to each other).
fn cross_section_basis(axis: Vec3, hint: Vec3) -> (Vec3, Vec3) {
    let a = axis.normalized().unwrap_or(Vec3::Z);
    let mut u = hint.cross(a);
    if u.length() < 1e-6 {
        u = Vec3::X.cross(a);
    }
    if u.length() < 1e-6 {
        u = Vec3::Y.cross(a);
    }
    let u = u.normalized().unwrap_or(Vec3::X);
    (u, a.cross(u))
}

/// Tapered limb from `a` to `b`. `profile` is `(t along the limb, radius)`.
fn tube(
    faces: &mut Vec<Face>,
    a: Vec3,
    b: Vec3,
    profile: &[(f64, f64)],
    hint: Vec3,
    color: Color32,
) {
    let axis = b - a;
    let len = axis.length();
    if len < 1e-9 || profile.len() < 2 {
        return;
    }
    let (u, v) = cross_section_basis(axis, hint);
    let ring = |k: usize, i: usize| {
        let (t, r) = profile[k];
        let angle = std::f64::consts::TAU * (i % SIDES) as f64 / SIDES as f64;
        a + axis.scale(t) + (u.scale(angle.cos()) + v.scale(angle.sin())).scale(r)
    };
    for k in 0..profile.len() - 1 {
        for i in 0..SIDES {
            let on_axis = a + axis.scale((profile[k].0 + profile[k + 1].0) * 0.5);
            quad(
                faces,
                [
                    ring(k, i),
                    ring(k, i + 1),
                    ring(k + 1, i + 1),
                    ring(k + 1, i),
                ],
                on_axis,
                color,
                false,
            );
        }
    }
    // Flat end caps: a fan at each end.
    for (k, end) in [(0, a), (profile.len() - 1, b)] {
        for i in 0..SIDES {
            quad(
                faces,
                [ring(k, i), ring(k, i + 1), end, end],
                a + axis.scale(0.5),
                color,
                false,
            );
        }
    }
}

fn sphere(faces: &mut Vec<Face>, centre: Vec3, radius: f64, color: Color32) {
    const LAT: usize = 5;
    let point = |lat: usize, lon: usize| {
        let phi = std::f64::consts::PI * lat as f64 / LAT as f64;
        let theta = std::f64::consts::TAU * (lon % SIDES) as f64 / SIDES as f64;
        centre
            + Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()).scale(radius)
    };
    for lat in 0..LAT {
        for lon in 0..SIDES {
            quad(
                faces,
                [
                    point(lat, lon),
                    point(lat, lon + 1),
                    point(lat + 1, lon + 1),
                    point(lat + 1, lon),
                ],
                centre,
                color,
                false,
            );
        }
    }
}

/// Where the hand sits: wrist point and the three hand axes.
struct HandFrame {
    wrist: Vec3,
    /// Along the fingers.
    f: Vec3,
    /// Toward the thumb side.
    t: Vec3,
    /// Out of the palm.
    n: Vec3,
}

impl HandFrame {
    fn at(&self, along: f64, across: f64, out: f64) -> Vec3 {
        self.wrist + self.f.scale(along) + self.t.scale(across) + self.n.scale(out)
    }
}

/// Tapered box between two stations along the hand: `(along)`, `(width)`, `(thickness)`.
#[allow(clippy::too_many_arguments)]
fn hand_block(
    faces: &mut Vec<Face>,
    h: &HandFrame,
    (a0, a1): (f64, f64),
    (w0, w1): (f64, f64),
    (h0, h1): (f64, f64),
    palm: Color32,
    back: Color32,
    side: Color32,
) {
    let centre = h.at((a0 + a1) * 0.5, 0.0, 0.0);
    let corner = |end: usize, sx: f64, sy: f64| {
        let (a, w, th) = if end == 0 { (a0, w0, h0) } else { (a1, w1, h1) };
        h.at(a, sx * w * 0.5, sy * th * 0.5)
    };
    // Palm (+n), back (-n), both sides, fingertip end.
    quad(
        faces,
        [
            corner(0, 1.0, 1.0),
            corner(0, -1.0, 1.0),
            corner(1, -1.0, 1.0),
            corner(1, 1.0, 1.0),
        ],
        centre,
        palm,
        false,
    );
    quad(
        faces,
        [
            corner(0, 1.0, -1.0),
            corner(0, -1.0, -1.0),
            corner(1, -1.0, -1.0),
            corner(1, 1.0, -1.0),
        ],
        centre,
        back,
        false,
    );
    for sx in [1.0, -1.0] {
        quad(
            faces,
            [
                corner(0, sx, 1.0),
                corner(0, sx, -1.0),
                corner(1, sx, -1.0),
                corner(1, sx, 1.0),
            ],
            centre,
            side,
            false,
        );
    }
    quad(
        faces,
        [
            corner(1, 1.0, 1.0),
            corner(1, -1.0, 1.0),
            corner(1, -1.0, -1.0),
            corner(1, 1.0, -1.0),
        ],
        centre,
        side,
        false,
    );
}

fn hand(faces: &mut Vec<Face>, sk: &ArmSkeleton) {
    let h = HandFrame {
        wrist: sk.wrist,
        f: sk.forearm_dir,
        t: sk.thumb_dir,
        n: sk.palm_normal,
    };
    // Palm, then fingers (narrower and flatter, like a mitten).
    hand_block(
        faces,
        &h,
        (0.0, PALM_LENGTH),
        (PALM_WIDTH * 0.85, PALM_WIDTH),
        (PALM_THICKNESS, PALM_THICKNESS * 0.9),
        PALM,
        BACK_OF_HAND,
        HAND_SIDE,
    );
    hand_block(
        faces,
        &h,
        (PALM_LENGTH, HAND_VISUAL_LENGTH),
        (PALM_WIDTH, PALM_WIDTH * 0.8),
        (PALM_THICKNESS * 0.9, PALM_THICKNESS * 0.6),
        PALM,
        BACK_OF_HAND,
        HAND_SIDE,
    );
    // Thumb: a short tapered finger on the thumb side only, angled forward.
    let base = h.at(0.02, PALM_WIDTH * 0.42, 0.0);
    let tip = h.at(0.115, PALM_WIDTH * 0.5 + 0.075, PALM_THICKNESS * 0.1);
    tube(
        faces,
        base,
        tip,
        &[(0.0, 0.026), (0.6, 0.024), (1.0, 0.019)],
        h.n,
        HAND_SIDE,
    );
    sphere(faces, tip, 0.019, HAND_SIDE);

    // Palm markings: two creases, a centre dot and finger separations.
    let lift = PALM_THICKNESS * 0.5 + 0.0015;
    let mark = |faces: &mut Vec<Face>, a: f64, across: f64, len: f64, wid: f64, out: f64| {
        quad(
            faces,
            [
                h.at(a, across - wid, out),
                h.at(a, across + wid, out),
                h.at(a + len, across + wid, out),
                h.at(a + len, across - wid, out),
            ],
            h.at(a, across, 0.0),
            PALM_LINE,
            true,
        );
    };
    mark(faces, 0.035, 0.0, 0.006, PALM_WIDTH * 0.42, lift);
    mark(faces, 0.075, 0.0, 0.006, PALM_WIDTH * 0.36, lift * 0.98);
    mark(faces, 0.05, 0.0, 0.014, 0.014, lift);
    for across in [-0.034, 0.0, 0.034] {
        mark(
            faces,
            PALM_LENGTH,
            across,
            FINGER_LENGTH * 0.92,
            0.0025,
            lift * 0.9,
        );
    }
    // Back of hand: knuckle dots.
    for across in [-0.045, -0.015, 0.015, 0.045] {
        let out = -(PALM_THICKNESS * 0.45 + 0.0015);
        quad(
            faces,
            [
                h.at(0.095, across - 0.008, out),
                h.at(0.095, across + 0.008, out),
                h.at(0.111, across + 0.008, out),
                h.at(0.111, across - 0.008, out),
            ],
            h.at(0.1, across, 0.0),
            KNUCKLE,
            true,
        );
    }
}

/// The whole arm: sleeve on the upper arm, skin on the forearm, hand.
pub fn arm_faces(sk: &ArmSkeleton) -> Vec<Face> {
    let mut faces = Vec::with_capacity(400);
    // Hint keeps the cross-section orientation stable between frames.
    let hint = Vec3::Y;
    // Upper arm: sleeve for the top part, skin below, with a slight muscle bulge.
    let mid = sk.shoulder + (sk.elbow - sk.shoulder).scale(0.55);
    tube(
        &mut faces,
        sk.shoulder,
        mid,
        &[(0.0, 0.085), (0.5, 0.088), (1.0, 0.078)],
        hint,
        SLEEVE,
    );
    tube(
        &mut faces,
        mid,
        sk.elbow,
        &[(0.0, 0.078), (0.5, 0.07), (1.0, 0.064)],
        hint,
        SKIN,
    );
    // Forearm tapers toward the wrist.
    tube(
        &mut faces,
        sk.elbow,
        sk.wrist,
        &[(0.0, 0.066), (0.25, 0.07), (1.0, 0.04)],
        hint,
        SKIN,
    );
    sphere(&mut faces, sk.shoulder, 0.095, SLEEVE);
    sphere(&mut faces, sk.elbow, 0.068, SKIN);
    sphere(&mut faces, sk.wrist, 0.042, SKIN);
    hand(&mut faces, sk);
    faces
}

/// What the palm is facing in the body frame (x right, y forward, z up).
pub fn palm_facing(palm_normal: Vec3) -> &'static str {
    if palm_normal.z > 0.6 {
        "up"
    } else if palm_normal.z < -0.6 {
        "down"
    } else {
        "sideways"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arm3d::skeleton;
    use crate::clinical::Side;

    fn arm(upper: Vec3, forearm: Vec3, pronation: f64, side: Side) -> ArmSkeleton {
        skeleton(upper, Some(forearm), Some(pronation), side)
    }

    #[test]
    fn the_arm_is_built_from_many_lit_faces_with_unit_normals() {
        let faces = arm_faces(&arm(Vec3::DOWN, Vec3::DOWN, 0.0, Side::Right));
        assert!(faces.len() > 250, "{}", faces.len());
        for f in &faces {
            assert!((f.n.length() - 1.0).abs() < 1e-9);
            assert!(
                f.v.iter()
                    .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
            );
        }
    }

    #[test]
    fn limbs_have_the_requested_thickness_around_their_axis() {
        let mut faces = Vec::new();
        tube(
            &mut faces,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            &[(0.0, 0.08), (1.0, 0.04)],
            Vec3::Y,
            SKIN,
        );
        assert!(faces.len() >= SIDES * 3);
        for f in &faces {
            for p in f.v {
                let radial = (p.x * p.x + p.y * p.y).sqrt();
                let allowed = 0.08 + (0.04 - 0.08) * (-p.z).clamp(0.0, 1.0);
                assert!(radial <= allowed + 1e-9, "{p:?}");
            }
        }
        // Side faces point away from the axis.
        for f in faces.iter().filter(|f| f.n.z.abs() < 0.5) {
            let radial = Vec3::new(f.center().x, f.center().y, 0.0);
            assert!(f.n.dot(radial) > 0.0);
        }
    }

    #[test]
    fn degenerate_limbs_produce_no_faces_instead_of_nan() {
        let mut faces = Vec::new();
        tube(
            &mut faces,
            Vec3::ZERO,
            Vec3::ZERO,
            &[(0.0, 0.1), (1.0, 0.1)],
            Vec3::Y,
            SKIN,
        );
        tube(
            &mut faces,
            Vec3::ZERO,
            Vec3::X,
            &[(0.0, 0.1)],
            Vec3::Y,
            SKIN,
        );
        assert!(faces.is_empty());
    }

    /// Faces of the hand's palm and back colours seen from `eye`.
    fn visible_tones(sk: &ArmSkeleton, eye: Vec3) -> (usize, usize) {
        let faces = arm_faces(sk);
        let count = |c: Color32| {
            faces
                .iter()
                .filter(|f| !f.decal && f.color == c && f.front_facing(eye))
                .count()
        };
        (count(PALM), count(BACK_OF_HAND))
    }

    #[test]
    fn palm_and_back_of_hand_are_different_faces_visible_from_opposite_sides() {
        for side in [Side::Right, Side::Left] {
            // Forearm forward, pronated: palm down.
            let sk = arm(Vec3::Y, Vec3::Y, 90.0, side);
            assert_eq!(palm_facing(sk.palm_normal), "down");
            let above = sk.wrist + Vec3::new(0.0, 0.0, 3.0);
            let below = sk.wrist - Vec3::new(0.0, 0.0, 3.0);
            let (palm_above, back_above) = visible_tones(&sk, above);
            let (palm_below, back_below) = visible_tones(&sk, below);
            assert!(
                back_above > 0 && palm_above == 0,
                "{side:?}: from above only the back shows"
            );
            assert!(
                palm_below > 0 && back_below == 0,
                "{side:?}: from below only the palm shows"
            );
        }
    }

    #[test]
    fn pronation_and_supination_flip_which_side_faces_up() {
        for side in [Side::Right, Side::Left] {
            let eye = Vec3::new(0.0, 0.4, 3.0);
            let supinated = arm(Vec3::Y, Vec3::Y, -90.0, side);
            let pronated = arm(Vec3::Y, Vec3::Y, 90.0, side);
            assert_eq!(palm_facing(supinated.palm_normal), "up");
            assert_eq!(palm_facing(pronated.palm_normal), "down");
            let (palm_up, back_up) = visible_tones(&supinated, eye);
            assert!(palm_up > 0 && back_up == 0, "{side:?}");
            let (palm_down, back_down) = visible_tones(&pronated, eye);
            assert!(back_down > 0 && palm_down == 0, "{side:?}");
        }
    }

    #[test]
    fn palm_face_normal_matches_the_estimated_palm_direction() {
        for side in [Side::Right, Side::Left] {
            for p in [-60.0, 0.0, 45.0] {
                let sk = arm(Vec3::DOWN, Vec3::DOWN, p, side);
                let faces = arm_faces(&sk);
                let palm: Vec<_> = faces
                    .iter()
                    .filter(|f| !f.decal && f.color == PALM)
                    .collect();
                assert!(!palm.is_empty());
                assert!(
                    palm.iter().all(|f| f.n.dot(sk.palm_normal) > 0.9),
                    "{side:?} {p}"
                );
                let back: Vec<_> = faces
                    .iter()
                    .filter(|f| !f.decal && f.color == BACK_OF_HAND)
                    .collect();
                assert!(
                    back.iter().all(|f| f.n.dot(sk.palm_normal) < -0.9),
                    "{side:?} {p}"
                );
            }
        }
    }

    #[test]
    fn the_thumb_sits_on_the_thumb_side_only() {
        for side in [Side::Right, Side::Left] {
            let sk = arm(Vec3::DOWN, Vec3::DOWN, 0.0, side);
            let faces = arm_faces(&sk);
            let reach = |dir: Vec3| {
                faces
                    .iter()
                    .filter(|f| f.color == HAND_SIDE)
                    .flat_map(|f| f.v)
                    .map(|p| (p - sk.wrist).dot(dir))
                    .fold(f64::MIN, f64::max)
            };
            assert!(reach(sk.thumb_dir) > PALM_WIDTH * 0.5 + 0.05);
            assert!(
                reach(-sk.thumb_dir) < PALM_WIDTH * 0.5 + 0.01,
                "{side:?}: nothing on the other side"
            );
        }
    }

    #[test]
    fn the_hand_follows_the_forearm_axis_and_ends_at_the_visual_length() {
        let sk = arm(Vec3::X, Vec3::Y, 20.0, Side::Right);
        let faces = arm_faces(&sk);
        let far = faces
            .iter()
            .filter(|f| f.color == PALM || f.color == BACK_OF_HAND)
            .flat_map(|f| f.v)
            .map(|p| (p - sk.wrist).dot(sk.forearm_dir))
            .fold(f64::MIN, f64::max);
        assert!((far - HAND_VISUAL_LENGTH).abs() < 1e-9, "{far}");
        const {
            assert!(HAND_VISUAL_LENGTH < crate::arm3d::HAND_LENGTH + crate::arm3d::BLADE_LENGTH);
        }
    }

    #[test]
    fn no_face_has_an_edge_longer_than_the_part_it_belongs_to() {
        // A stray long edge would show up as a spike across the 3D view.
        use crate::armpose::ArmState;
        use crate::pnf3d::{PnfConfig, PnfPattern};
        for pattern in PnfPattern::ALL {
            for side in [Side::Right, Side::Left] {
                let cfg = PnfConfig {
                    reach: 1.0,
                    ..PnfConfig::new(pattern, side)
                };
                for i in 0..=20 {
                    let st: ArmState = cfg.reference_state(f64::from(i) / 20.0, true);
                    let sk = st.skeleton(side);
                    for f in arm_faces(&sk) {
                        for k in 0..4 {
                            let edge = (f.v[k] - f.v[(k + 1) % 4]).length();
                            // Limb quads span a segment section; the longest legitimate edge is
                            // along the upper arm or forearm (about half an arm length).
                            assert!(
                                edge < 0.45,
                                "{pattern:?} {side:?} step {i}: edge {edge} color {:?}",
                                f.color
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn palm_facing_labels() {
        assert_eq!(palm_facing(Vec3::Z), "up");
        assert_eq!(palm_facing(-Vec3::Z), "down");
        assert_eq!(palm_facing(Vec3::X), "sideways");
        assert_eq!(palm_facing(Vec3::new(0.0, 0.0, 0.6)), "sideways");
    }

    #[test]
    fn nonsense_skeletons_never_produce_nan_faces() {
        let sk = skeleton(Vec3::ZERO, Some(Vec3::ZERO), Some(f64::NAN), Side::Left);
        for f in arm_faces(&sk) {
            assert!(
                f.v.iter()
                    .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
            );
            assert!(f.n.x.is_finite());
        }
    }
}
