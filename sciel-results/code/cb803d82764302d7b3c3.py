"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIME_LIMIT = 110.0  # Safety margin before 120s kill


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _time_exceeded():
    return (time.time() - _get_start_time()) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in the TSP tour.
    Uses a vectorized greedy heuristic with a 1-step lookahead penalty.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if _time_exceeded():
        # Fallback to plain nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_remaining = len(unvisited_nodes)
    
    # 1. Distance from current to all candidates
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from each candidate to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Distance from each candidate to the *next* nearest unvisited node (1-step lookahead)
    # Compute pairwise distances among unvisited nodes using np.ix_ for correct 2D submatrix
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity so we don't pick self
    np.fill_diagonal(sub_dist, np.inf)
    
    # Min distance to any other unvisited node
    d_next_min = np.min(sub_dist, axis=1)
    
    # Score calculation:
    # We want to minimize: dist(current, i) + weight1 * dist(i, next_best) + weight2 * dist(i, dest)
    # This approximates the immediate cost plus a penalty for leaving the tour in a bad state
    # (far from other nodes) and a penalty for being far from the destination (if we are late).
    
    # Weights:
    # As n_remaining decreases, the destination penalty should increase.
    # As n_remaining is large, the "connectivity" penalty (d_next_min) is more important to avoid clusters.
    
    # Simple adaptive weights:
    # alpha for d_next_min: 0.2 (constant, helps avoid isolated nodes)
    # beta for d_dest: increases as we get closer to the end
    beta = 0.2 + 0.3 * (1.0 - n_remaining / 50.0)
    
    scores = d_current + 0.2 * d_next_min + beta * d_dest
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
