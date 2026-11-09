import numpy as np
import time


def _nearest_neighbor_tour(distance_matrix):
    """Generate a tour using greedy nearest neighbor starting at node 0."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    for _ in range(n - 1):
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def _two_opt(tour, distance_matrix):
    """Improve tour using 2-opt."""
    n = len(tour)
    improved = True
    tour = list(tour)
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment between i+1 and j
                old_cost = distance_matrix[tour[i], tour[(i + 1) % n]] + \
                           distance_matrix[tour[j], tour[(j + 1) % n]]
                new_cost = distance_matrix[tour[i], tour[j]] + \
                           distance_matrix[tour[(i + 1) % n], tour[(j + 1) % n]]
                if new_cost < old_cost - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # Generate reference tour
    nn_tour = _nearest_neighbor_tour(distance_matrix)
    opt_tour = _two_opt(nn_tour, distance_matrix)
    
    # Build set of edges in the optimized tour (including closing edge)
    tour_edges = set()
    for i in range(n):
        u = opt_tour[i]
        v = opt_tour[(i + 1) % n]
        tour_edges.add((u, v))
        tour_edges.add((v, u))
    
    # Compute base heuristic: 1/distance
    # Avoid division by zero
    safe_dist = distance_matrix + 1e-9
    base_heuristic = 1.0 / safe_dist
    
    # Boost edges in the reference tour
    heuristic = base_heuristic.copy()
    for (u, v) in tour_edges:
        heuristic[u, v] = base_heuristic[u, v] * 10.0
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
