use std::{
    cell::RefCell,
    cmp::{max, min, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    rc::Rc,
};

use itertools::Itertools;
use nalgebra as na;
use ordered_float::OrderedFloat;

pub type EdgeIndex = [usize; 2];
pub type Edge = [na::Vector2<f64>; 2];
pub type TriangleIndex = [usize; 3];
pub type Triangle = [na::Vector2<f64>; 3];

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
pub fn min_spanning_by_prim(points: &Vec<na::Vector2<f64>>) -> Vec<Edge> {
    if points.is_empty() {
        return Vec::new();
    }

    // calculate the edge length
    let edge_lengths: Vec<Vec<f64>> = points
        .iter()
        .map(|p1| points.iter().map(|p2| p1.metric_distance(&p2)).collect())
        .collect();

    // indicies of the edges in the spanning tree
    let mut graph: Vec<EdgeIndex> = Vec::new();

    // visited vertices
    let mut visited_verticies = HashSet::from([0]);
    // remaining edges
    let mut heap_edges: BinaryHeap<_> = edge_lengths[0]
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, len)| -> Reverse<(OrderedFloat<f64>, [usize; 2])> {
            Reverse((OrderedFloat(*len), [0, i]))
        })
        .collect();

    while visited_verticies.len() < points.len() {
        let Reverse((_, [i, j])) = heap_edges.pop().unwrap();
        if visited_verticies.contains(&j) {
            // skip if the edge is already visited
            continue;
        }

        graph.push([i, j]);
        visited_verticies.insert(j);
        for (k, len) in edge_lengths[j].iter().enumerate() {
            if !visited_verticies.contains(&k) {
                heap_edges.push(Reverse((OrderedFloat(*len), [j, k])));
            }
        }
    }

    // convert indicies to edge
    graph
        .into_iter()
        .map(|[i, j]| [points[i], points[j]])
        .collect()
}

/// Enumerate edges of a triangle
///
/// # Arguments
///
/// - `triangle`: Triangle
///
/// # Returns
///
/// - `Vec<[i, j]>`: Edges in triangle
#[inline]
pub fn enumerate_edges(triangle: &TriangleIndex) -> Vec<EdgeIndex> {
    triangle
        .iter()
        .zip(triangle.iter().cycle().skip(1))
        .map(|(&i, &j)| [min(i, j), max(i, j)])
        .collect()
}

/// Find a graph that minimizes the total edge length, using a heuristic approach
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
#[derive(Default)]
pub struct TriangleSpanningCalculator {
    points: Vec<na::Vector2<f64>>,
    /// points number
    n: usize,
    /// indicies of the edges in the spanning tree
    graph: HashSet<EdgeIndex>,
    /// edge length
    edge_lengths: Vec<Vec<f64>>,
    /// remaining all triangles with current adding cost
    all_triangles: Vec<Rc<RefCell<(f64, TriangleIndex)>>>,
    /// remaining all triangles with current adding cost, referrenced by edge
    all_triangles_on_edges: HashMap<EdgeIndex, Vec<Rc<RefCell<(f64, TriangleIndex)>>>>,

    /// next candidate edges; belong to only 1 triangle
    open_edges: Vec<EdgeIndex>,
    triangles_in_graphs_on_edges: HashMap<EdgeIndex, Vec<TriangleIndex>>,

    /// `Some(answer)` if arleady calculated
    answer: Option<Vec<Edge>>,
}

impl TriangleSpanningCalculator {
    /// Create a new [`TriangleSpanningCalculator`]
    ///
    /// # Arguments
    ///
    /// - `points`: vertices of the graph
    fn new(points: &Vec<na::Vector2<f64>>) -> Self {
        let mut x = Self {
            ..Default::default()
        };
        x.initialize(points);
        x
    }

    /// Initialize the data structure for calculating
    ///
    /// # Arguments
    ///
    /// - `points`: vertices of the graph
    fn initialize(&mut self, points: &Vec<na::Vector2<f64>>) {
        self.points = points.clone();
        self.n = self.points.len();
        self.edge_lengths = self
            .points
            .iter()
            .map(|p1| points.iter().map(|p2| p1.metric_distance(&p2)).collect())
            .collect();

        self.all_triangles = (0..self.n)
            .combinations(3)
            .map(|ps| {
                Rc::new(RefCell::new((
                    self.edge_lengths[ps[0]][ps[1]]
                        + self.edge_lengths[ps[1]][ps[2]]
                        + self.edge_lengths[ps[2]][ps[0]],
                    [ps[0], ps[1], ps[2]],
                )))
            })
            .collect();

        self.all_triangles_on_edges = {
            let mut m: HashMap<EdgeIndex, Vec<Rc<RefCell<(f64, TriangleIndex)>>>> = HashMap::new();
            for t in &self.all_triangles {
                let [i, j, k] = t.borrow().1;
                m.entry([i, j]).or_default().push(t.clone());
                m.entry([i, k]).or_default().push(t.clone());
                m.entry([j, k]).or_default().push(t.clone());
            }
            m
        };
    }

    #[inline]
    fn sort_to_edge(x: usize, y: usize) -> EdgeIndex {
        [min(x, y), max(x, y)]
    }

    #[inline]
    fn sort_to_triangle(x: usize, y: usize, z: usize) -> TriangleIndex {
        let mut v = vec![x, y, z];
        v.sort();
        [v[0], v[1], v[2]]
    }

    /// Build the initial triangle of graph.
    ///
    /// This mostly return the values,
    /// but updates the value `self.open_edges` directly.
    ///
    /// # Returns
    ///
    /// - `(new_edges, new_vertices, next_triangle)`
    ///   - `new_edges`: new edges to add
    ///   - `new_vertices`: new vertices to add
    ///   - `next_triangle`: new triangle to add
    fn build_initial_triangle(&mut self) -> (Vec<EdgeIndex>, Vec<usize>, TriangleIndex) {
        let new_edges: Vec<EdgeIndex>;
        let new_vertices: Vec<usize>;
        let next_triangle: TriangleIndex;
        // select the smallest triangle
        next_triangle = self
            .all_triangles
            .iter()
            .map(|rc| rc.borrow())
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
            .1;
        new_edges = enumerate_edges(&next_triangle);
        new_vertices = next_triangle.to_vec();
        // add all new edges
        self.open_edges = new_edges.clone();

        (new_edges, new_vertices, next_triangle)
    }

    /// Select the shortest total edge length triangle with `open_edges`
    ///
    /// # Arguments
    ///
    /// - `visited`: vertices already added
    ///
    /// # Returns
    ///
    /// - shortest triangle
    fn select_shortest_triangle_with_open_edges(&self, visited: &HashSet<usize>) -> TriangleIndex {
        self.open_edges
            .iter()
            .map(|e| -> Vec<_> {
                self.all_triangles_on_edges[&[e[0], e[1]]]
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
            .1
    }

    /// Build the second or later triangle of graph.
    ///
    /// This mostly return the values,
    /// but updates the value `self.open_edges` directly.
    ///
    /// # Arguments
    ///
    /// - `visited`: vertices already added
    ///
    /// # Returns
    ///
    /// - `(new_edges, new_vertices, next_triangle)`
    ///   - `new_edges`: new edges to add
    ///   - `new_vertices`: new vertices to add
    ///   - `next_triangle`: new triangle to add
    fn expand_graph_with_next_triangle(
        &mut self,
        visited: &HashSet<usize>,
    ) -> (Vec<EdgeIndex>, Vec<usize>, TriangleIndex) {
        // select the next triangle
        let next_triangle: TriangleIndex = self.select_shortest_triangle_with_open_edges(visited);

        // select new vertices among the vertices in the new triangle
        let (old_v, new_v): ([usize; 2], usize) = {
            let (o, n): (Vec<usize>, Vec<usize>) =
                next_triangle.iter().partition(|v| visited.contains(&v));
            assert_eq!(n.len(), 1, "n: {:?}", n);
            (o.try_into().unwrap(), n[0])
        };

        let new_edges: Vec<EdgeIndex> = vec![
            Self::sort_to_edge(old_v[0], new_v),
            Self::sort_to_edge(old_v[1], new_v),
        ];
        let new_vertices: Vec<usize> = vec![new_v];

        // add new edges and remove old edges
        self.open_edges.retain(|&vs| vs != old_v);
        self.open_edges.extend(&new_edges);

        (new_edges, new_vertices, next_triangle)
    }

    /// Build the graph as the initial answer.
    /// This updates the value `self.graph`.
    ///
    /// This also updates the values `self.open_edges`
    /// and `self.triangles_in_graphs_on_edges
    fn construct_initial_graph(&mut self) {
        // visited vertices
        let mut visited: HashSet<usize> = HashSet::new();

        while visited.len() < self.n {
            let new_edges: Vec<EdgeIndex>;
            let new_vertices: Vec<usize>;
            let next_triangle: TriangleIndex;
            (new_edges, new_vertices, next_triangle) = if self.graph.is_empty() {
                // first time
                self.build_initial_triangle()
            } else {
                // not first time
                self.expand_graph_with_next_triangle(&visited)
            };

            let next_edges: Vec<EdgeIndex> = enumerate_edges(&next_triangle);
            next_edges.iter().for_each(|&e| {
                self.triangles_in_graphs_on_edges
                    .entry(e)
                    .or_default()
                    .push(next_triangle.clone());
            });

            self.graph.extend(&new_edges);
            new_vertices.into_iter().for_each(|v| {
                visited.insert(v);
            });

            for e in next_edges {
                for t in self.all_triangles_on_edges[&[e[0], e[1]]].clone() {
                    t.borrow_mut().0 -= self.edge_lengths[e[0]][e[1]];
                }
            }
        }
    }

    #[inline]
    fn get_triangle_num_with_edge(&self, e: &EdgeIndex) -> usize {
        self.triangles_in_graphs_on_edges
            .get(e)
            .map_or(0, |ts| ts.len())
    }

    #[inline]
    fn filter_from_two<T, F>(arr: [T; 2], f: F) -> Option<T>
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        match (f(&arr[0]), f(&arr[1])) {
            (true, false) => Some(arr[0].clone()),
            (false, true) => Some(arr[1].clone()),
            _ => None,
        }
    }

    fn select_new_edge_for_open_triangle_with_an_open_edge(
        &self,
        edge_del: &EdgeIndex,
        open_edge_adj: &EdgeIndex,
    ) -> Option<EdgeIndex> {
        // If the open edges are connected as a -- b -- c -- d,
        // and a -- c are also directly connected,
        // then remove the edge a -- b and connect b -- d.
        let candidate: EdgeIndex = {
            // pivot: point in `open_edges_adj`, not in `edge_del`
            // v: the other point in `open_edges_adj`
            let (pivot, v): (usize, usize) = if edge_del.contains(&open_edge_adj[0]) {
                (open_edge_adj[1], open_edge_adj[0])
            } else {
                (open_edge_adj[0], open_edge_adj[1])
            };

            (0..self.n)
                .find_map(|i| {
                    let e: EdgeIndex = Self::sort_to_edge(i, pivot);
                    if (self.get_triangle_num_with_edge(&e) == 1) && !(&e == open_edge_adj) {
                        Some(Self::sort_to_edge(i, v))
                    } else {
                        None
                    }
                })
                .unwrap()
        };
        // cannot connect if candidate is longer than `e_del`
        if self.edge_lengths[candidate[0]][candidate[1]]
            >= self.edge_lengths[edge_del[0]][edge_del[1]]
        {
            return None;
        }
        Some(candidate)
    }

    fn walk_to_find_adjacent_vertex_with_same_side(
        &self,
        pivot: usize,
        adj_vs_open: [usize; 2],
        adj_vs_closed: [usize; 2],
    ) -> usize {
        // start with `adj_vs_open[0]`
        let mut i: usize = adj_vs_open[0];
        let mut i_prev: usize = pivot;

        // find one of the `adj_vs_closed` which is in same side
        loop {
            // find next adjacent vertex along the open edge
            let i_next: usize = (0..self.n)
                .find(|&j| {
                    let e: EdgeIndex = Self::sort_to_edge(i, j);
                    (self.get_triangle_num_with_edge(&e) == 1) && (j != i_prev)
                })
                .unwrap();
            i_prev = i;
            i = i_next;

            if adj_vs_closed.contains(&i) {
                break;
            }
        }

        i
    }

    /// For the vertex connected with 2 open edges and 2 closed edges,
    /// partition the 4 vertices connected with it to 2 groups made of 2 verticies by position.
    ///
    /// # Arguments
    ///
    /// - `pivot`: The vertex connected with 2 open edges and 2 closed edges
    /// - `adj_vs_open`: The 2 vertices connected with `pivot` by the open edge
    /// - `adj_vs_closed`: The 2 vertices connected with `pivot` by the closed edge
    ///
    /// # Returns
    ///
    /// - `(a, b)`
    ///   - `a`: The 2 vertices in the same side
    ///   - `b`: The 2 vertices in the other side
    fn partition_vertices_by_position(
        &self,
        pivot: usize,
        adj_vs_open: [usize; 2],
        adj_vs_closed: [usize; 2],
    ) -> ([usize; 2], [usize; 2]) {
        // adjacent vertices in same side when remove `e_del`
        let a: [usize; 2] = {
            [
                adj_vs_open[0],
                self.walk_to_find_adjacent_vertex_with_same_side(pivot, adj_vs_open, adj_vs_closed),
            ]
        };

        // the other vertices not `a`
        let b: [usize; 2] = [
            Self::filter_from_two(adj_vs_open, |&x| !a.contains(&x)).unwrap(),
            Self::filter_from_two(adj_vs_closed, |&x| !a.contains(&x)).unwrap(),
        ];
        (a, b)
    }

    fn select_new_edge_for_open_triangle_without_open_edge(
        &self,
        edge_del: &EdgeIndex,
        closed_edges_adj: [EdgeIndex; 2],
    ) -> Option<EdgeIndex> {
        let candidate: EdgeIndex = {
            // common point of closed_edges_adj
            let pivot: usize =
                Self::filter_from_two(closed_edges_adj[0], |&x| closed_edges_adj[1].contains(&x))
                    .unwrap();
            // adjacent vertices of `pivot`, with open edge between `pivot`
            let adj_vs_open: [usize; 2] = (0..self.n)
                .filter(|&i| {
                    let e: EdgeIndex = Self::sort_to_edge(i, pivot);
                    self.get_triangle_num_with_edge(&e) == 1
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            // adjacent vertices of `pivot`, with closed edge between `pivot`
            let adj_vs_closed: [usize; 2] = [
                Self::filter_from_two(closed_edges_adj[0], |&x| x != pivot).unwrap(),
                Self::filter_from_two(closed_edges_adj[1], |&x| x != pivot).unwrap(),
            ];

            let (a, b): ([usize; 2], [usize; 2]) =
                self.partition_vertices_by_position(pivot, adj_vs_open, adj_vs_closed);

            // find shortest edge between `a` and `b`
            a.into_iter()
                .zip(b.into_iter())
                .map(|(x, y)| Self::sort_to_edge(x, y))
                .min_by(|e1, e2| {
                    self.edge_lengths[e1[0]][e1[1]]
                        .partial_cmp(&self.edge_lengths[e2[0]][e2[1]])
                        .unwrap()
                })
                .unwrap()
        };

        // If `e_del` is already the longest edge between `a` and `b`,
        // it cannot be connected.
        if candidate == *edge_del {
            return None;
        }
        Some(candidate)
    }

    fn select_new_edge_for_open_triangle(&self, edge_del: &EdgeIndex) -> Option<EdgeIndex> {
        // remaining edges in triangle with `e_del`, classified by open-closed
        let (open_edges_adj, closed_edges_adj): (Vec<EdgeIndex>, Vec<EdgeIndex>) = {
            assert_eq!(
                self.get_triangle_num_with_edge(edge_del),
                1,
                "[{:?}]: {:?}",
                edge_del,
                self.triangles_in_graphs_on_edges
                    .get(edge_del)
                    .unwrap_or(&Vec::new())
            );
            let t: TriangleIndex = self.triangles_in_graphs_on_edges[edge_del][0];
            enumerate_edges(&t)
                .into_iter()
                .filter(|e| e != edge_del)
                .partition(|e| self.get_triangle_num_with_edge(e) == 1)
        };

        if open_edges_adj.len() == 2 {
            // cannot swap
            return None;
        }

        if open_edges_adj.len() == 1 {
            dbg!("open-oc");
            self.select_new_edge_for_open_triangle_with_an_open_edge(edge_del, &open_edges_adj[0])
        } else {
            // when no more open edges is in the triangle
            dbg!("open-cc");
            self.select_new_edge_for_open_triangle_without_open_edge(
                edge_del,
                closed_edges_adj.try_into().unwrap(),
            )
        }
    }

    fn select_new_edge_for_closed_triangle(&self, edge_del: &EdgeIndex) -> Option<EdgeIndex> {
        // if `edge_del` is closed
        // When `edge_del` is x -- y,
        // and a -- x -- b -- y -- a is connected in this order,
        // the remove x -- y and connect a -- b.

        // vertices connected with the ends of `e_del`
        dbg!(&edge_del);
        let vs: Vec<usize> = self.triangles_in_graphs_on_edges[edge_del]
            .iter()
            .flatten()
            .cloned()
            .filter(|p| !edge_del.contains(p))
            .collect();
        assert_eq!(vs.len(), 2, "vs: {:?}", vs);

        let e = Self::sort_to_edge(vs[0], vs[1]);
        // cannot connect if candidate is longer than `e_del`,
        // or candidate is already connected
        if self.graph.contains(&e) {
            return None;
        }
        if self.edge_lengths[e[0]][e[1]] >= self.edge_lengths[edge_del[0]][edge_del[1]] {
            return None;
        }

        Some(e)
    }

    fn swap_edges(&mut self, edge_del: &EdgeIndex, edge_add: &EdgeIndex) {
        // swap edges
        self.graph.retain(|e| e != edge_del);
        self.graph.insert(edge_add.clone());
        self.open_edges.retain(|e| e != edge_del);
        self.open_edges.push(edge_add.clone());

        // remove triangles with `e1` from `triangles_in_graphs_on_edges`
        for t in self.triangles_in_graphs_on_edges[edge_del].clone() {
            for e in &enumerate_edges(&t) {
                if e == edge_del {
                    continue;
                }

                self.triangles_in_graphs_on_edges
                    .get_mut(e)
                    .unwrap()
                    .retain(|&tt| tt != t);
            }
        }
        self.triangles_in_graphs_on_edges.remove(edge_del);

        // add triangles with `e2` to `triangles_in_graphs_on_edges`
        for i in 0..self.n {
            let ee1: EdgeIndex = Self::sort_to_edge(i, edge_add[0]);
            let ee2: EdgeIndex = Self::sort_to_edge(i, edge_add[1]);
            if !self.graph.contains(&ee1) || !self.graph.contains(&ee2) {
                continue;
            }

            let t: TriangleIndex = Self::sort_to_triangle(i, edge_add[0], edge_add[1]);
            dbg!(t);

            self.triangles_in_graphs_on_edges
                .entry(ee1)
                .or_default()
                .push(t);
            self.triangles_in_graphs_on_edges
                .entry(ee2)
                .or_default()
                .push(t);
            self.triangles_in_graphs_on_edges
                .entry(edge_add.clone())
                .or_default()
                .push(t);
        }

        dbg!(format!("swaped {:?}, {:?}", &edge_del, &edge_add));
    }

    fn optimize_graph_edges(&mut self) -> bool {
        // edges of graph in descending order
        let edges_desc: Vec<EdgeIndex> = self
            .graph
            .iter()
            .cloned()
            .sorted_by(|a, b| {
                self.edge_lengths[b[0]][b[1]]
                    .partial_cmp(&self.edge_lengths[a[0]][a[1]])
                    .unwrap()
            })
            .collect();

        for edge_del in &edges_desc {
            let maybe_edge_add: Option<EdgeIndex> = {
                // if `edge_del` is open triangle or not
                let is_open = self.get_triangle_num_with_edge(edge_del) == 1;
                if is_open {
                    self.select_new_edge_for_open_triangle(edge_del)
                } else {
                    dbg!("closed");
                    self.select_new_edge_for_closed_triangle(edge_del)
                }
            };
            if let Some(edge_add) = maybe_edge_add {
                self.swap_edges(edge_del, &edge_add);
                return true;
            }
        }

        false
    }

    /// Calculate the answer
    ///
    /// # Returns
    ///
    /// - `Vec<[p0, p1]>`: the edges of the graph
    fn calc_answer(&mut self) -> Vec<Edge> {
        if self.n <= 1 {
            return Vec::new();
        }
        if self.n == 2 {
            return vec![[self.points[0], self.points[1]]];
        }

        self.construct_initial_graph();

        dbg!(&self.graph);
        dbg!(&self.triangles_in_graphs_on_edges);
        // swap open_edges if graph gets shorter
        loop {
            if !self.optimize_graph_edges() {
                break;
            }
        }

        // convert indicies to edge
        self.graph
            .iter()
            .map(|&[i, j]| [self.points[i], self.points[j]])
            .collect()
    }

    /// Solve the problem
    ///
    /// When `solve()` is already called, it use stored answer.
    ///
    /// # Returns
    ///
    /// - `Vec<[p0, p1]>`: the edges of the graph
    pub fn solve(&mut self) -> Vec<Edge> {
        if self.answer.is_none() {
            self.answer = Some(self.calc_answer());
        }

        self.answer.clone().unwrap()
    }
}

/// Find a graph that minimizes the total edge length, using a heuristic approach
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
pub fn small_triangular_spanning(points: &Vec<na::Vector2<f64>>) -> Vec<Edge> {
    TriangleSpanningCalculator::new(points).solve()
}

pub(crate) mod test {
    use super::*;

    use rand::{thread_rng, Rng};
    use scad_tree::prelude::*;

    /// Generate random points
    ///
    /// # Arguments
    ///
    /// - `n`: number of points
    ///
    /// # Returns
    ///
    /// - [`Vec<na::Vector2<f64>>`]: random points
    fn generate_random_points(n: usize) -> Vec<na::Vector2<f64>> {
        let mut rng = thread_rng();
        (0..n)
            .map(|_| {
                (0..5)
                    .map(|_| {
                        na::Vector2::new(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0))
                    })
                    .fold(na::Vector2::zeros(), |a, b| 1.2 * a + b)
            })
            .collect::<Vec<_>>()
    }

    fn generate_test_object(points: &Vec<na::Vector2<f64>>, edges: &Vec<Edge>) -> Scad {
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
        Scad {
            op: ScadOp::Union,
            children: shapes,
        }
    }

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

        // random points
        let points: Vec<na::Vector2<f64>> = generate_random_points(n);

        // small triangular mesh
        let edges: Vec<Edge> = small_triangular_spanning(&points);

        Some(generate_test_object(&points, &edges))
    }
}
