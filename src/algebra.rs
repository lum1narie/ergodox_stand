use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

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
