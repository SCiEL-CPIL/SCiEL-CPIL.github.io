"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if _time_exceeded():
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small remaining sets, use exact permutation search
    if n_unvisited <= 8:
        best_cost = float('inf')
        best_first = -1
        U = unvisited_nodes
        for p in permutations(U):
            cost = distance_matrix[current_node, p[0]]
            for i in range(len(p) - 1):
                cost += distance_matrix[p[i], p[i+1]]
            cost += distance_matrix[p[-1], destination_node]
            if cost < best_cost:
                best_cost = cost
                best_first = p[0]
        return int(best_first)
    
    # For larger sets, use vectorized nearest-neighbor gap metric with adaptive weighting
    U = unvisited_nodes
    m = n_unvisited
    
    d_curr = distance_matrix[current_node, U]
    d_dest = distance_matrix[U, destination_node]
    
    # Build sub-matrix of distances between unvisited nodes
    sub = distance_matrix[np.ix_(U, U)].copy()
    np.fill_diagonal(sub, np.inf)
    
    # Nearest-neighbor gap metric:
    # For each candidate v, compute the distance to its nearest other unvisited node
    nearest_unvisited_dist = np.min(sub, axis=1)
    
    # Baseline: distance from current to the nearest unvisited node
    baseline = np.min(d_curr)
    
    # Gap score: how much extra cost does picking v add locally?
    gap_scores = d_curr + 0.3 * nearest_unvisited_dist - baseline
    
    # 2-step lookahead: min over v in U\{u} of (d(u,v) + d(v,dest))
    # Vectorized: add d_dest to each column, then take min over columns
    two_step = np.min(sub + d_dest[np.newaxis, :], axis=1)
    
    # Adaptive weights based on progress through the tour
    # Early in tour: weight immediate cost and gap more
    # Late in tour: weight exit cost more
    progress = max(0.0, (50.0 - m) / 50.0)
    
    w_curr = 1.0
    w_gap = 0.3 * (1.0 - 0.5 * progress)
    w_2step = 0.2 * (1.0 - progress)
    w_dest = 0.3 + 0.7 * progress
    
    scores = w_curr * d_curr + w_gap * nearest_unvisited_dist + w_2step * two_step + w_dest * d_dest
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])
# EVOLVE-BLOCK-END
