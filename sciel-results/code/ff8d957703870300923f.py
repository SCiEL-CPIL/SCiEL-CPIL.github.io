"""Improved constructive TSP heuristic: return-aware nearest neighbour with adaptive weighting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a return-aware greedy heuristic with adaptive weighting.
    
    The scoring combines:
    1. Forward distance (current -> candidate)
    2. Return distance (candidate -> destination) with adaptive weight
    
    The weight on return distance increases as the tour nears completion,
    helping to close the loop effectively.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: nearest neighbor
        unv = np.asarray(unvisited_nodes)
        if unv.size == 0:
            return int(destination_node)
        if unv.size == 1:
            return int(unv[0])
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    dm = distance_matrix
    
    # Forward cost: distance from current to each candidate
    d_forward = dm[current_node, unv]
    # Return cost: distance from each candidate to destination
    d_return = dm[unv, destination_node]
    
    # Adaptive weight: as fewer nodes remain, weight the return more heavily
    # Total nodes in the tour is roughly the original size; we can estimate it
    # The weight should transition smoothly from low (early) to high (late)
    # Use a smooth sigmoid-like transition based on fraction of tour completed
    n_cities = distance_matrix.shape[0]
    frac_completed = 1.0 - (n / n_cities)
    
    # Smooth transition: w_return goes from ~0.3 to ~1.0
    # Using a tanh-like smooth step
    # At frac_completed=0 (start): w_return ~ 0.3
    # At frac_completed=1 (end): w_return ~ 1.0
    # Smooth step: w = 0.3 + 0.7 * (2*frac_completed - frac_completed^2)
    # Or simpler: linear interpolation with clamping
    w_return = 0.3 + 0.7 * frac_completed
    w_return = np.clip(w_return, 0.3, 1.0)
    
    # Total score: forward + w_return * return
    score = d_forward + w_return * d_return
    
    best_idx = np.argmin(score)
    return int(unv[best_idx])
# EVOLVE-BLOCK-END
