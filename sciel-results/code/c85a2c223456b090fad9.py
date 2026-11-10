"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Heuristic combining:
    1. Inverse distance base (power 2)
    2. Edges from a NN+2Opt reference tour boosted
    3. k-NN edges (k=4) boosted
    4. Crossing-edge penalty for long-range edges
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]

    # 1. Nearest Neighbor tour
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node

    # 2. 2-Opt improvement (limited)
    def tour_length(t):
        idx = np.array(t)
        return np.sum(distance_matrix[idx, np.roll(idx, -1)])

    current_len = tour_length(tour)
    improved = True
    max_iter = 300
    iter_count = 0

    while improved and iter_count < max_iter:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                if new < old - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    current_len -= old - new
                    improved = True
                    if time.time() > deadline:
                        break
        if time.time() > deadline:
            break

    # 3. Build heuristic matrix
    safe_dist = np.where(distance_matrix == 0, 1e-9, distance_matrix)
    base_heuristic = 1.0 / (safe_dist ** 2)

    # Boost reference tour edges
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        base_heuristic[u, v] *= 12.0
        base_heuristic[v, u] *= 12.0

    # Boost k-NN edges (k=4)
    k = 4
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        for j in nn_indices:
            base_heuristic[i, j] *= 2.5

    # Penalty for long-range edges that cross the tour
    # Identify tour edges as a set for quick lookup
    tour_edge_set = set()
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        tour_edge_set.add((min(u, v), max(u, v)))

    # Penalize edges not in tour and not in k-NN that are relatively long
    median_dist = np.median(safe_dist[safe_dist > 0]) if np.any(safe_dist > 0) else 1.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (min(i, j), max(i, j)) in tour_edge_set:
                continue
            if safe_dist[i, j] > 1.5 * median_dist:
                base_heuristic[i, j] *= 0.5

    base_heuristic = np.maximum(base_heuristic, 1e-9)
    return base_heuristic
# EVOLVE-BLOCK-END
