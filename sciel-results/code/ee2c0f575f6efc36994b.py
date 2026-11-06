"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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

    # If time is running out, fall back to nearest neighbor for speed
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Structural change: density-aware heuristic
    # Instead of pairwise lookahead, consider the local topology of unvisited nodes.
    # Nodes in sparse regions are preferred to avoid trapping the tour in dense clusters.
    # Score = dist(current, candidate) / (local_density + eps)
    # where local_density = mean distance from candidate to other unvisited nodes.
    
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute sub-matrix of distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Local density: average distance to other unvisited nodes
    # Subtract diagonal contribution (self-distance = 0)
    local_density = sub_matrix.sum(axis=1) / (n_unvisited - 1)
    
    # Normalize to avoid division issues; use sqrt to dampen extreme values
    scores = dist_from_current / (np.sqrt(local_density) + 1e-6)
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
