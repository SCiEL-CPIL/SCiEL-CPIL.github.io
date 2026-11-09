"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # Compute distances from current to all candidates
    dist_to_candidate = distance_matrix[current_node, candidates]
    
    # For small candidate sets, use a weighted lookahead that considers returning to destination
    # The weight increases as we get closer to the end of the tour
    if n <= 15:
        dist_candidate_to_dest = distance_matrix[candidates, destination_node]
        # Weight ramps up from 0.3 to 1.0 as n goes from 15 to 1
        weight = 0.3 + 0.7 * (15.0 - n) / 14.0
        scores = dist_to_candidate + weight * dist_candidate_to_dest
        return int(candidates[np.argmin(scores)])
    
    # For medium-sized sets, use a lighter lookahead
    if n <= 30:
        dist_candidate_to_dest = distance_matrix[candidates, destination_node]
        weight = 0.3
        scores = dist_to_candidate + weight * dist_candidate_to_dest
        return int(candidates[np.argmin(scores)])
    
    # For large sets (early in the tour), use nearest neighbor
    # This is fast and avoids over-committing to a specific direction too early
    return int(candidates[np.argmin(dist_to_candidate)])
# EVOLVE-BLOCK-END
