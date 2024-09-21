mod algebra;

use std::fs::create_dir;

use itertools::iproduct;
use nalgebra as na;
use once_cell::sync::Lazy;
use scad_tree::prelude::*;

const BOTTOM_PLATE_HEIGHT: f64 = 3.5;
const BOTTOM_PLATE_SIZE: f64 = 8.;

const FULCRUM_X_ON_ERGODOX: f64 = 130.;
const FULCRUM_Y_ON_ERGODOX: f64 = 100.;

// vector which is perpendicular to ergodox
const UPPER_VECTOR: Lazy<na::Vector3<f64>> = Lazy::new(|| na::Vector3::<f64>::new(-2., 1.19, 9.52));

const FULCRUM_BUTTOM_R: f64 = 8.;
const FULCRUM_TOP_CYLINDER_R: f64 = 4.;
const FULCRUM_TOP_SPHERE_R: f64 = 6.;

const BASE_HEIGHT: f64 = 10.;

const CORNER_SUPPORT_THICKNESS: f64 = 7.;
const CORNER_SUPPORT_SIZE: f64 = 25.;

/// Generate the rotation matrix of the upper vector
///
/// # Returns
///
/// 3D rotation matrix
#[inline]
fn rot_matrix() -> na::Rotation3<f64> {
    na::Rotation3::rotation_between(&na::Vector3::<f64>::z(), &UPPER_VECTOR.normalize()).unwrap()
}

/// Generate the euler angles of the rotation matrix in degrees
///
/// # Parameters
///
/// - `r`: 3D rotation matrix
///
/// # Returns
///
/// Euler angles `[x, y, z]`
#[inline]
fn euler_angle_degrees(r: &na::Rotation3<f64>) -> [f64; 3] {
    let (rot_x, rot_y, rot_z) = r.euler_angles();
    [rot_x.to_degrees(), rot_y.to_degrees(), rot_z.to_degrees()]
}

/// Generate the basic shape of the Ergodox
///
/// # Returns
///
/// Basic shape of the Ergodox
#[inline]
fn ergodox_shape(height: f64) -> Scad {
    mirror!(
        [0.,1.,0.],
        union!(
            // body
            cube!([158., 136., height]);
            // thumb
            translate!(
                [
                158. -  (48. * 64.0_f64.to_radians().sin()),
                86. -  (48. * 64.0_f64.to_radians().cos()),
                0.
                ],
                rotate!(
                    [0., 0., 23.],
                    cube!([96., 70., height]);
                );
            );
        );
    )
}

/// Generate the filled shape of the top corner support
///
/// # Returns
///
/// Filled shape of the top corner support
fn top_corner_support_filled() -> Scad {
    // generate the vertices
    let v: Pt3s = {
        let verticies_original: Vec<na::Vector3<f64>> = {
            let xs = [-CORNER_SUPPORT_THICKNESS, CORNER_SUPPORT_SIZE];
            let ys = [-CORNER_SUPPORT_SIZE, CORNER_SUPPORT_THICKNESS];
            let zs = [-CORNER_SUPPORT_THICKNESS, CORNER_SUPPORT_SIZE];

            iproduct!(zs, ys, xs)
                .map(|(z, y, x)| na::Vector3::new(x, y, z))
                .collect()
        };
        let verticies: Vec<na::Vector3<f64>> = verticies_original
            .iter()
            .map(|v| rot_matrix() * v + na::Vector3::new(0., 0., BASE_HEIGHT))
            .collect();
        let base_points: Vec<na::Vector3<f64>> = verticies
            .iter()
            .map(|p| na::Vector3::new(p.x, p.y, 0.))
            .collect();
        Pt3s::from_pt3s(
            verticies
                .into_iter()
                .chain(base_points.into_iter())
                .map(|p| Pt3::new(p.x, p.y, p.z))
                .collect::<Vec<_>>(),
        )
    };
    // generate the indices of the faces
    let f: Vec<Vec<u64>> = {
        let vertical: Vec<Vec<u64>> = [[0, 1], [1, 3], [3, 7], [7, 6], [6, 4], [4, 0]]
            .into_iter()
            .map(|[a, b]| -> Vec<u64> { vec![a, b, b + 8, a + 8] })
            .collect();
        let cube_side: Vec<Vec<u64>> = [[0, 1], [1, 3]]
            .into_iter()
            .map(|[a, b]| -> Vec<u64> { vec![a, a + 4, b + 4, b] })
            .collect();
        let top: Vec<u64> = vec![4, 6, 7, 5];
        let bottom: Vec<u64> = vec![8, 9, 11, 15, 14, 12];

        // merge all
        vec![vertical, cube_side, vec![top], vec![bottom]].concat()
    };

    polyhedron!(
        v,
        Faces::from_faces(
            f.into_iter()
                .map(|idx| Indices::from_indices(idx))
                .collect::<Vec<_>>()
        )
    )
}

/// Generate the tip points of the fulcrums
///
/// # Returns
///
/// Tip points of the fulcrums
fn fulcrums_points() -> Vec<na::Vector3<f64>> {
    let rot_matrix = rot_matrix();

    // based on the Ergodox before rotation
    let fulcrums_original = vec![
        na::Vector3::new(0., -FULCRUM_Y_ON_ERGODOX, 0.),
        na::Vector3::new(158.0 * 0.3, -120.0, 0.), // TODO:
        na::Vector3::new(FULCRUM_X_ON_ERGODOX, -FULCRUM_Y_ON_ERGODOX, 0.),
        na::Vector3::new(FULCRUM_X_ON_ERGODOX * 0.7, 0., 0.),
    ];

    // rotate the points
    fulcrums_original
        .into_iter()
        .map(|p| rot_matrix * p)
        .collect::<Vec<_>>()
}

/// Generate the fulcrums
///
/// # Returns
///
/// Fulcrums
fn fulcrums() -> Vec<Scad> {
    fulcrums_points()
        .into_iter()
        .map(|p| {
            union!(
                // cylinder body
                translate!(
                    [p.x, p.y, 0.],
                    cylinder!(h=p.z + BASE_HEIGHT, r1=FULCRUM_BUTTOM_R, r2=FULCRUM_TOP_CYLINDER_R);
                );
                // sphere on the tip
                translate!(
                    [p.x, p.y, p.z + BASE_HEIGHT],
                    sphere!(r=FULCRUM_TOP_SPHERE_R);
                );
            )
        })
        .collect()
}

/// Generate the Ergodox stand for the left hand
///
/// # Returns
///
/// Ergodox stand for the left hand
fn ergodox_stand_left() -> Scad {
    let rot_matrix = rot_matrix();
    let [rot_x, rot_y, rot_z] = euler_angle_degrees(&rot_matrix);

    // shape of the Ergodox to cut the stand
    let ergodox_rotated = rotate!(
        [rot_x, rot_y, rot_z],
        ergodox_shape(100.);
    );

    // supporting shapes
    let support_shapes: Vec<Scad> = vec![vec![top_corner_support_filled()], fulcrums()].concat();

    // base plate
    let base = {
        // base points to connect
        let base_points: Vec<na::Vector2<f64>> = {
            // base of supports
            let ps = vec![
                vec![na::Vector2::new(0., 0.)],
                fulcrums_points()
                    .into_iter()
                    .map(|p| na::Vector2::new(p.x, p.y))
                    .collect(),
            ]
            .concat();

            vec![
                ps.clone(),
                // the weighted average of the `ps`
                vec![ps.iter().fold(na::Vector2::zeros(), |a, b| a + b) / (ps.len() as f64)],
            ]
            .concat()
            // ps
        };
        dbg!(&base_points);

        // edges of the base to connect all of `base_points` with triangle
        let base_edges: Vec<[na::Vector2<f64>; 2]> =
            algebra::small_triangular_spanning(&base_points);

        Scad {
            op: ScadOp::Union,
            children: base_edges
                .into_iter()
                .map(|[p1, p2]| {
                    // connecting edge between the two points
                    hull!(
                        linear_extrude!(
                            BOTTOM_PLATE_HEIGHT,
                            translate!(
                                [p1.x, p1.y, 0.],
                                circle!(d=BOTTOM_PLATE_SIZE);
                            );
                        );
                        linear_extrude!(
                            BOTTOM_PLATE_HEIGHT,
                            translate!(
                                [p2.x, p2.y, 0.],
                                circle!(d=BOTTOM_PLATE_SIZE);
                            );
                        );
                    )
                })
                .collect(),
        }
    };

    // stand without cut
    let filled = Scad {
        op: ScadOp::Union,
        children: vec![support_shapes, vec![base]].concat(),
    };

    // cut with Ergodox shape
    difference!(
        filled;
        translate!(
            [0.,0.,BASE_HEIGHT],
            ergodox_rotated;
        );
    )
}

/// Generate the Ergodox stand for the right hand
///
/// # Returns
///
/// Ergodox stand for the right hand
#[inline]
fn ergodox_stand_right() -> Scad {
    mirror!(
        [1., 0., 0.],
        ergodox_stand_left();
    )
}

fn main() {
    let el = ergodox_stand_left();
    let er = ergodox_stand_right();

    let _ = create_dir("things");
    el.save("things/ergodox_stand_left.scad");
    er.save("things/ergodox_stand_right.scad");

    let tri_test = algebra::test::test_small_triangular_spanning(30).unwrap();
    tri_test.save("things/tri_test.scad");
}
