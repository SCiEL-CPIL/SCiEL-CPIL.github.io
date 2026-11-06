"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIME_LIMIT = 115.0  # Safety margin before 120s kill


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _time_exceeded():
    return (time.time() - _get_start_time()) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if _time_exceeded():
        # Fallback to plain nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Return-aware scoring: balance distance to next node vs distance from next to destination
    d_to_next = distance_matrix[current_node, unvisited_nodes]
    d_next_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive alpha: when few nodes remain, weight the return leg more heavily
    n_remaining = len(unvisited_nodes)
    # alpha goes from ~0.5 (many nodes, prioritize nearest) to ~1.0 (few nodes, prioritize closure)
    alpha = 0.5 + 0.5 * (1.0 - n_remaining / 50.0)
    
    scores = d_to_next + alpha * d_next_to_dest
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
