use std::{
    cell::RefCell,
    cmp::{max, min, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    rc::Rc,
};

use itertools::Itertools;
use nalgebra as na;
use ordered_float::OrderedFloat;

/// Calculates the minimum spanning tree by Prim's algorithm
///
/// # Arguments
///
/// - `points`: Points to be connected as a graph
///
/// # Returns
///
/// Minimum spanning tree `vec![[p0, p1], ...]`,
/// which is consisted by the edges (p0, p1), ...
pub fn min_spanning_by_prim(points: &Vec<na::Vector2<f64>>) -> Vec<[na::Vector2<f64>; 2]> {
    if points.is_empty() {
        return Vec::new();
    }

    // calculate the edge length
    let edges: Vec<Vec<f64>> = points
        .iter()
        .map(|p1| points.iter().map(|p2| p1.metric_distance(&p2)).collect())
        .collect();

    // indicies of the edges in the spanning tree
    let mut graph: Vec<[usize; 2]> = Vec::new();

    // visited vertices
    let mut visited = HashSet::from([0]);
    // remaining edges
    let mut heap: BinaryHeap<_> = edges[0]
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, len)| -> Reverse<(OrderedFloat<f64>, [usize; 2])> {
            Reverse((OrderedFloat(*len), [0, i]))
        })
        .collect();

    while visited.len() < points.len() {
        let Reverse((_, [i, j])) = heap.pop().unwrap();
        if visited.contains(&j) {
            // skip if the edge is already visited
            continue;
        }

        graph.push([i, j]);
        visited.insert(j);
        for (k, len) in edges[j].iter().enumerate() {
            if !visited.contains(&k) {
                heap.push(Reverse((OrderedFloat(*len), [j, k])));
            }
        }
    }

    // convert indicies to edge
    graph
        .into_iter()
        .map(|[i, j]| [points[i], points[j]])
        .collect()
}

/// Find a graph that minimizes the total edge length
/// , using a heuristic approach
/// while satisfying the following conditions:
///
/// - Edges are generated as follows:
///   - From all possible combinations of 3 points on the graph,
///     select (v-2) sets, and connect all edges of the resulting triangles.
/// - All vertices must be connected.
/// - Each edge can belong to a maximum of two triangles.
///   - There are exactly two edges per vertex that belong to only one triangle.
///     - These edges are called open edges.
///     - Edges that belong to two triangles are called closed edges.
/// - Note: Cases where the 3 points are collinear are not considered.
///
/// # Arguments
///
/// - `points`: vertices of the graph
///
/// # Returns
///
/// - `Vec<[p0, p1]>`: the edges of the graph
pub fn small_triangular_spanning(points: &Vec<na::Vector2<f64>>) -> Vec<[na::Vector2<f64>; 2]> {
    if points.len() <= 1 {
        return Vec::new();
    }
    if points.len() == 2 {
        return vec![[points[0], points[1]]];
    }

    // calculate the edge length
    let edges: Vec<Vec<f64>> = points
        .iter()
        .map(|p1| points.iter().map(|p2| p1.metric_distance(&p2)).collect())
        .collect();

    // remaining triangles having current adding cost
    let all_triangles: Vec<Rc<RefCell<(f64, [usize; 3])>>> = (0..points.len())
        .combinations(3)
        .map(|v| {
            Rc::new(RefCell::new((
                edges[v[0]][v[1]] + edges[v[1]][v[2]] + edges[v[2]][v[0]],
                [v[0], v[1], v[2]],
            )))
        })
        .collect();

    let all_triangles_on_edges: HashMap<[usize; 2], Vec<Rc<RefCell<(f64, [usize; 3])>>>> = {
        let mut m: HashMap<[usize; 2], Vec<Rc<RefCell<(f64, [usize; 3])>>>> = HashMap::new();
        for t in &all_triangles {
            let [i, j, k] = t.borrow().1;
            m.entry([i, j]).or_default().push(t.clone());
            m.entry([i, k]).or_default().push(t.clone());
            m.entry([j, k]).or_default().push(t.clone());
        }
        m
    };

    // indicies of the edges in the spanning tree
    let mut graph: HashSet<[usize; 2]> = HashSet::new();

    // visited vertices
    let mut visited: HashSet<usize> = HashSet::new();
    // next candidate edges; belong to only 1 triangle
    let mut open_edges: Vec<[usize; 2]> = Vec::new();

    let mut triangles_in_graphs_on_edges: HashMap<[usize; 2], Vec<[usize; 3]>> = HashMap::new();

    while visited.len() < points.len() {
        let new_edges: Vec<[usize; 2]>;
        let new_vertices: Vec<usize>;
        let next_triangle: [usize; 3];
        if graph.is_empty() {
            // first time
            // select the smallest triangle
            next_triangle = all_triangles
                .iter()
                .map(|rc| rc.borrow())
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap()
                .1;
            new_edges = next_triangle
                .iter()
                .zip(next_triangle.iter().cycle().skip(1))
                .map(|(&a, &b)| [min(a, b), max(a, b)])
                .collect();
            new_vertices = next_triangle.to_vec();
            // add all new edges
            open_edges = new_edges.clone();
        } else {
            // not first time
            // select the next triangle
            next_triangle = open_edges
                .iter()
                .map(|e| {
                    all_triangles_on_edges[&[e[0], e[1]]]
                        .iter()
                        .filter(|rc| {
                            rc.borrow()
                                .1
                                .iter()
                                .all(|&v| (v == e[0]) || (v == e[1]) || !visited.contains(&v))
                        })
                        .collect::<Vec<_>>()
                })
                .flatten()
                .map(|rc| rc.borrow().clone())
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap()
                .1;

            // select new vertices among the vertices in the new triangle
            let (old_v, new_v) = {
                let i = next_triangle
                    .iter()
                    .position(|v| !visited.contains(v))
                    .unwrap();
                let js = (0..=2).filter(|&j| j != i).collect::<Vec<_>>();
                (
                    [next_triangle[js[0]], next_triangle[js[1]]],
                    next_triangle[i],
                )
            };

            new_edges = vec![
                [min(old_v[0], new_v), max(old_v[0], new_v)],
                [min(old_v[1], new_v), max(old_v[1], new_v)],
            ];
            new_vertices = vec![new_v];

            // add new edges and remove old edges
            open_edges.retain(|&vs| vs != old_v);
            open_edges.extend(&new_edges);
        }

        let next_edges: Vec<[usize; 2]> = next_triangle
            .iter()
            .zip(next_triangle.iter().cycle().skip(1))
            .map(|(&a, &b)| [min(a, b), max(a, b)])
            .collect();
        next_edges.iter().for_each(|&e| {
            triangles_in_graphs_on_edges
                .entry(e)
                .or_default()
                .push(next_triangle.clone());
        });

        graph.extend(&new_edges);
        new_vertices.into_iter().for_each(|v| {
            visited.insert(v);
        });

        for e in next_edges {
            for t in all_triangles_on_edges[&[e[0], e[1]]].clone() {
                t.borrow_mut().0 -= edges[e[0]][e[1]];
            }
        }
    }

    // swap open_edges if graph gets shorter
    loop {
        // edges of graph in descending order
        let edges_desc: Vec<[usize; 2]> = graph
            .iter()
            .cloned()
            .sorted_by(|a, b| edges[b[0]][b[1]].partial_cmp(&edges[a[0]][a[1]]).unwrap())
            .collect();

        let mut is_moved = false;
        for e_del in &edges_desc {
            // if `e_del` is open triangle or not
            let is_open = triangles_in_graphs_on_edges[e_del].len() == 1;

            let e_add: [usize; 2] = if is_open {
                // remaining edges in triangle with `e_del`, classified by open-closed
                let (open_edges_adj, closed_edges_adj): (Vec<[usize; 2]>, Vec<[usize; 2]>) = {
                    let mut o: Vec<[usize; 2]> = Vec::new();
                    let mut c: Vec<[usize; 2]> = Vec::new();

                    let vs: [usize; 3] = triangles_in_graphs_on_edges[e_del][0];
                    let es: Vec<[usize; 2]> = vs
                        .iter()
                        .zip(vs.iter().cycle().skip(1))
                        .map(|(&a, &b)| [min(a, b), max(a, b)])
                        .collect();

                    for e in es {
                        if &e == e_del {
                            continue;
                        }
                        if triangles_in_graphs_on_edges[&e].len() == 1 {
                            o.push(e);
                        } else {
                            c.push(e);
                        }
                    }

                    (o, c)
                };

                if open_edges_adj.len() == 2 {
                    // cannot swap
                    continue;
                }

                if open_edges_adj.len() == 1 {
                    // If the open edges are connected as a -- b -- c -- d,
                    // and a -- c are also directly connected,
                    // then remove the edge a -- b and connect b -- d.
                    let candidate: [usize; 2] = {
                        // point in `open_edges_adj[0]`, not in `e1`
                        let pivot: usize = open_edges_adj[0]
                            .iter()
                            .find(|&v| !e_del.contains(v))
                            .copied()
                            .unwrap();
                        // the other point in `open_edges_adj[0]`
                        let v: usize = open_edges_adj[0]
                            .iter()
                            .find(|&x| x != &pivot)
                            .copied()
                            .unwrap();

                        (0..points.len())
                            .find_map(|i| {
                                let e: [usize; 2] = [min(i, pivot), max(i, pivot)];
                                if triangles_in_graphs_on_edges
                                    .get(&e)
                                    .map_or(0, |ts| ts.len())
                                    == 1
                                {
                                    if e == open_edges_adj[0] {
                                        None
                                    } else {
                                        Some([min(i, v), max(i, v)])
                                    }
                                } else {
                                    None
                                }
                            })
                            .unwrap()
                    };
                    // cannot connect if candidate is longer than `e_del`
                    if edges[candidate[0]][candidate[1]] >= edges[e_del[0]][e_del[1]] {
                        continue;
                    }
                    candidate
                } else {
                    // when no more open edges is in the triangle
                    let candidate: [usize; 2] = {
                        // common point of closed_edges_adj
                        let pivot: usize = closed_edges_adj[0]
                            .iter()
                            .find(|&v| closed_edges_adj[1].contains(v))
                            .copied()
                            .unwrap();
                        // adjacent vertices of `pivot`, with open edge between `pivot`
                        let adj_vs_open: Vec<usize> = (0..points.len())
                            .filter_map(|i| {
                                let e: [usize; 2] = [min(i, pivot), max(i, pivot)];
                                if triangles_in_graphs_on_edges
                                    .get(&e)
                                    .map_or(0, |ts| ts.len())
                                    == 1
                                {
                                    Some(i)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        // adjacent vertices of `pivot`, with closed edge between `pivot`
                        let adj_vs_closed: Vec<usize> = closed_edges_adj
                            .iter()
                            .flatten()
                            .cloned()
                            .filter(|&v| v != pivot)
                            .collect();

                        // adjacent vertices in same side when remove `e_del`
                        let a: Vec<usize> = {
                            // start with `adj_vs_open[0]`
                            let mut i: usize = adj_vs_open[0];
                            let mut i_prev: usize = pivot;

                            // find one of the `adj_vs_closed` which is in same side
                            loop {
                                // find next adjacent vertex along the open edge
                                let i_next: usize = (0..points.len())
                                    .find_map(|j| {
                                        let e: [usize; 2] = [min(j, i), max(j, i)];
                                        if triangles_in_graphs_on_edges
                                            .get(&e)
                                            .map_or(0, |ts| ts.len())
                                            == 1
                                        {
                                            if j == i_prev {
                                                None
                                            } else {
                                                Some(j)
                                            }
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap();
                                i_prev = i;
                                i = i_next;

                                if adj_vs_closed.contains(&i) {
                                    break;
                                }
                            }
                            vec![adj_vs_open[0], i]
                        };

                        // the other vertices not `a`
                        let b: Vec<usize> = vec![
                            adj_vs_open
                                .iter()
                                .find(|v| !a.contains(*v))
                                .copied()
                                .unwrap(),
                            adj_vs_closed
                                .iter()
                                .find(|v| !a.contains(*v))
                                .copied()
                                .unwrap(),
                        ];

                        // find shortest edge between `a` and `b`
                        a.into_iter()
                            .zip(b.into_iter())
                            .map(|(x, y)| -> [usize; 2] { [min(x, y), max(x, y)] })
                            .min_by(|e1, e2| {
                                edges[e1[0]][e1[1]]
                                    .partial_cmp(&edges[e2[0]][e2[1]])
                                    .unwrap()
                            })
                            .unwrap()
                    };

                    // If `e_del` is already the longest edge between `a` and `b`,
                    // it cannot be connected.
                    if &candidate == e_del {
                        continue;
                    }
                    candidate
                }
            } else {
                // if `e_del` is closed
                // When `e_del` is x -- y,
                // and a -- x -- b -- y -- a is connected in this order,
                // the remove x -- y and connect a -- b.

                // vertices connected with the ends of `e_del`
                let vs: Vec<usize> = triangles_in_graphs_on_edges[e_del]
                    .iter()
                    .flatten()
                    .cloned()
                    .filter(|p| (p != &e_del[0]) && (p != &e_del[1]))
                    .collect();
                assert_eq!(vs.len(), 2);

                let e = [min(vs[0], vs[1]), max(vs[0], vs[1])];
                // cannot connect if candidate is longer than `e_del`,
                // or candidate is already connected
                if graph.contains(&e) {
                    continue;
                }
                if edges[e[0]][e[1]] >= edges[e_del[0]][e_del[1]] {
                    continue;
                }

                e
            };

            // swap edges
            graph.retain(|e| e != e_del);
            graph.insert(e_add.clone());
            open_edges.retain(|e| e != e_del);
            open_edges.push(e_add.clone());

            // remove triangles with `e1` from `triangles_in_graphs_on_edges`
            for t in triangles_in_graphs_on_edges[e_del].clone() {
                for e in t
                    .iter()
                    .zip(t.iter().cycle().skip(1))
                    .map(|(&i, &j)| [min(i, j), max(i, j)])
                {
                    if &e == e_del {
                        continue;
                    }

                    triangles_in_graphs_on_edges
                        .get_mut(&e)
                        .unwrap()
                        .retain(|&tt| tt != t);
                }
            }
            triangles_in_graphs_on_edges.remove(e_del);

            // add triangles with `e2` to `triangles_in_graphs_on_edges`
            for i in 0..points.len() {
                let ee1: [usize; 2] = [min(i, e_add[0]), max(i, e_add[0])];
                let ee2: [usize; 2] = [min(i, e_add[1]), max(i, e_add[1])];
                if !graph.contains(&ee1) || !graph.contains(&ee2) {
                    continue;
                }

                let t: [usize; 3] = [i, e_add[0], e_add[1]]
                    .into_iter()
                    .sorted()
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                triangles_in_graphs_on_edges.entry(ee1).or_default().push(t);
                triangles_in_graphs_on_edges.entry(ee2).or_default().push(t);
                triangles_in_graphs_on_edges
                    .entry(e_add)
                    .or_default()
                    .push(t);
            }

            dbg!(format!("swaped {:?}, {:?}", &e_del, &e_add));

            is_moved = true;
            break;
        }

        // end if no more edges to swap
        if !is_moved {
            break;
        }
    }

    // convert indicies to edge
    graph
        .into_iter()
        .map(|[i, j]| [points[i], points[j]])
        .collect()
}

pub(crate) mod test {
    use super::*;

    use rand::{thread_rng, Rng};
    use scad_tree::prelude::*;

    /// Create a small spanning triangular mesh from the random points
    ///
    /// # Arguments
    ///
    /// - `n`: number of points
    ///
    /// # Returns
    ///
    /// - [`Some(Scad)`]: [`Scad`] object of test 3D object
    /// - `None`: if `n < 3`
    pub fn test_small_triangular_spanning(n: usize) -> Option<Scad> {
        if n < 3 {
            return None;
        }

        let mut rng = thread_rng();
        // random points
        let points: Vec<na::Vector2<f64>> = (0..n)
            .map(|_| {
                (0..5)
                    .map(|_| {
                        na::Vector2::new(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0))
                    })
                    .fold(na::Vector2::zeros(), |a, b| 1.2 * a + b)
            })
            .collect::<Vec<_>>();

        // small triangular mesh
        let edges: Vec<[na::Vector2<f64>; 2]> = small_triangular_spanning(&points);

        // pillars on the points
        let pillars = points.into_iter().map(|p| {
            mirror! (
                [0., 0., 1.],
                translate!(
                    [p.x, p.y, 0.],
                    cylinder!(h=10., r=4.);
                );
            )
        });

        // the edges of mesh
        let base_edges = edges.into_iter().map(|[p1, p2]| {
            hull! (
                translate!(
                    [p1.x, p1.y, 0.],
                    cylinder!(h=1., r=1.);
                );
                translate!(
                    [p2.x, p2.y, 0.],
                    cylinder!(h=1., r=1.);
                );
            )
        });

        // pillars + edges
        let shapes: Vec<Scad> = pillars.chain(base_edges).collect();
        Some(Scad {
            op: ScadOp::Union,
            children: shapes,
        })
    }
}
