mod algebra;

use std::{
    fs::{create_dir, File},
    io::Write as _,
};

use itertools::iproduct;
use nalgebra as na;
use once_cell::sync::Lazy;
use scadman::prelude::*;

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

const SMALL: f64 = 0.001;

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
pub fn ergodox_shape(height: f64) -> Mirror3D {
    Mirror3D::build_with(|mb| {
        mb.v([0., 1., 0.]).apply_to(any_scads3d![
            // body
            Cube::build_with(|cb| {
                cb.size([158., 136., height]);
            }),
            // thumb
            Translate3D::build_with(|tb| {
                tb.v([
                    158. - (48. * 64.0_f64.to_radians().sin()),
                    86. - (48. * 64.0_f64.to_radians().cos()),
                    0.,
                ])
                .apply_to(Rotate3D::build_with(|rb| {
                    rb.deg([0., 0., 23.]).apply_to(Cube::build_with(|cb| {
                        cb.size([96., 70., height]);
                    }));
                }));
            })
        ]);
    })
}

/// Generate the shape of the top corner foot of Ergodox EZ
///
/// # Returns
///
/// Shape of the top corner foot
#[inline]
fn ergodox_top_corner_foot_shape() -> Translate3D {
    Translate3D::build_with(|tb| {
        tb.v([17., -17.5, -3.]).apply_to(Cylinder::build_with(|cb| {
            cb.h(3. + SMALL).d(10.);
        }));
    })
}

/// Generate the filled shape of the top corner support
///
/// # Returns
///
/// Filled shape of the top corner support
fn top_corner_support_filled() -> Polyhedron {
    // generate the vertices
    let v: Vec<Point3D> = {
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
        [verticies, base_points].concat()
    };
    // generate the indices of the faces
    let f: Vec<Vec<usize>> = {
        let vertical: Vec<Vec<usize>> = [[0, 1], [1, 3], [3, 7], [7, 6], [6, 4], [4, 0]]
            .into_iter()
            .map(|[a, b]| -> Vec<usize> { vec![a, b, b + 8, a + 8] })
            .collect();
        let cube_side: Vec<Vec<usize>> = [[0, 1], [1, 3]]
            .into_iter()
            .map(|[a, b]| -> Vec<usize> { vec![a, a + 4, b + 4, b] })
            .collect();
        let top: Vec<usize> = vec![4, 6, 7, 5];
        let bottom: Vec<usize> = vec![8, 9, 11, 15, 14, 12];

        // merge all
        [vertical, cube_side, vec![top], vec![bottom]].concat()
    };

    Polyhedron::build_with(|pb| {
        pb.points(v).faces(f);
    })
}

/// Generate the tip points of the fulcrums
///
/// # Returns
///
/// Tip points of the fulcrums
#[inline]
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
#[inline]
fn fulcrums() -> Vec<Union3D> {
    fulcrums_points()
        .into_iter()
        .map(|p| {
            Union3D::build_with(|ub| {
                ub.apply_to(any_scads3d![
                    // cylinder body
                    Translate3D::build_with(|tb| {
                        tb.v([p.x, p.y, 0.]).apply_to(Cylinder::build_with(|cb| {
                            cb.h(p.z + BASE_HEIGHT)
                                .r([FULCRUM_BUTTOM_R, FULCRUM_TOP_CYLINDER_R]);
                        }));
                    }),
                    // sphere on the tip
                    Translate3D::build_with(|tb| {
                        tb.v([p.x, p.y, p.z + BASE_HEIGHT])
                            .apply_to(Sphere::build_with(|sb| {
                                sb.r(FULCRUM_TOP_SPHERE_R);
                            }));
                    })
                ]);
            })
        })
        .collect()
}

/// Generate the edges to connect the points
///
/// # Parameters
///
/// `points` - The points to connect
///
/// # Returns
///
/// [`Vec<Scad>`] include edges to connect the points
#[inline]
fn connect_points(points: &Vec<na::Vector2<f64>>) -> Vec<Hull3D> {
    // edges to connect all of `points` with triangle
    let edges: Vec<[na::Vector2<f64>; 2]> = algebra::small_triangular_spanning(points);

    edges
        .into_iter()
        .map(|[p1, p2]| {
            // connecting edge between the two points
            Hull3D::build_with(|hb| {
                hb.apply_to(any_scads3d![
                    LinearExtrude::build_with(|lb| {
                        lb.height(BOTTOM_PLATE_HEIGHT)
                            .apply_to(Translate2D::build_with(|tb| {
                                tb.v(p1).apply_to(Circle::build_with(|cb| {
                                    cb.d(BOTTOM_PLATE_SIZE);
                                }));
                            }));
                    }),
                    LinearExtrude::build_with(|lb| {
                        lb.height(BOTTOM_PLATE_HEIGHT)
                            .apply_to(Translate2D::build_with(|tb| {
                                tb.v(p2).apply_to(Circle::build_with(|cb| {
                                    cb.d(BOTTOM_PLATE_SIZE);
                                }));
                            }));
                    }),
                ]);
            })
        })
        .collect()
}

/// Generate the Ergodox stand for the left hand
///
/// # Returns
///
/// Ergodox stand for the left hand
pub fn ergodox_stand_left() -> Difference3D {
    let rot_matrix = rot_matrix();
    let rot_ed = euler_angle_degrees(&rot_matrix);

    // shape of the Ergodox to cut the stand
    let ergodox_rotated = Rotate3D::build_with(|rb| {
        rb.deg(rot_ed).apply_to(Union3D::build_with(|ub| {
            ub.apply_to(any_scads3d![
                ergodox_shape(100.),
                ergodox_top_corner_foot_shape(),
            ]);
        }));
    });

    // supporting shapes
    let support_shapes: Vec<Box<dyn ScadObject3D>> = [
        any_scads3d![top_corner_support_filled()],
        fulcrums()
            .into_iter()
            .map(|x| Box::new(x) as Box<dyn ScadObject3D>)
            .collect(),
    ]
    .concat();

    // base plate
    let base = {
        // base points to connect
        let base_points: Vec<na::Vector2<f64>> = {
            // base of supports
            let ps = [vec![na::Vector2::new(0., 0.)],
                fulcrums_points()
                    .into_iter()
                    .map(|p| na::Vector2::new(p.x, p.y))
                    .collect()]
            .concat();

            [ps.clone(),
                // the weighted average of the `ps`
                vec![ps.iter().fold(na::Vector2::zeros(), |a, b| a + b) / (ps.len() as f64)]]
            .concat()
            // ps
        };

        let base_objs: Vec<Hull3D> = connect_points(&base_points);

        Union3D::build_with(|ub| {
            ub.apply_to(
                base_objs
                    .into_iter()
                    .map(|x| Box::new(x) as Box<dyn ScadObject3D>)
                    .collect::<Vec<_>>(),
            );
        })
    };

    // stand without cut
    let filled = Union3D::build_with(|ub| {
        ub.apply_to(
            [
                support_shapes,
                vec![Box::new(base) as Box<dyn ScadObject3D>],
            ]
            .concat(),
        );
    });

    // cut with Ergodox shape
    Difference3D::build_with(|db| {
        db.apply_to(any_scads3d![
            filled,
            Translate3D::build_with(|tb| {
                tb.v([0., 0., BASE_HEIGHT]).apply_to(ergodox_rotated);
            }),
        ]);
    })
}

/// Generate the Ergodox stand for the right hand
///
/// # Returns
///
/// Ergodox stand for the right hand
#[inline]
fn ergodox_stand_right() -> Mirror3D {
    Mirror3D::build_with(|mb| {
        mb.v([1., 0., 0.]).apply_to(ergodox_stand_left());
    })
}

fn main() {
    let el = ergodox_stand_left();
    let er = ergodox_stand_right();

    let _ = create_dir("things");
    {
        let mut f = File::create("things/ergodox_stand_left.scad").unwrap();
        f.write_all(el.to_code().as_bytes()).unwrap();
    }
    {
        let mut f = File::create("things/ergodox_stand_right.scad").unwrap();
        f.write_all(er.to_code().as_bytes()).unwrap();
    }

    let tri_test = algebra::test::test_small_triangular_spanning(30).unwrap();
    {
        let mut f = File::create("things/tri_test.scad").unwrap();
        f.write_all(tri_test.to_code().as_bytes()).unwrap();
    }
}
