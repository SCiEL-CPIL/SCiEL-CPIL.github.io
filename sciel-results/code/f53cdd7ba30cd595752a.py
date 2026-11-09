"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Returns a modified distance matrix where frequently-used edges are penalized,
    encouraging the 2-opt search to explore different tour structures.
    """
    n = edge_distance.shape[0]
    
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = 300.0  # seconds, with safety margin
    
    # Scale penalty relative to the mean edge distance
    mean_dist = np.mean(edge_distance)
    
    # Base penalty weight - calibrated to be meaningful but not overwhelming
    penalty_weight = 0.25 * mean_dist
    
    # Create a copy of the distance matrix
    penalized_dist = edge_distance.copy()
    
    # Additive penalty based on historical usage
    # This is more stable than pure multiplicative penalties
    penalized_dist += penalty_weight * edge_n_used
    
    # Additional penalty for edges in the current local optimum tour
    # This specifically breaks the current tour structure
    tour_penalty_weight = 0.15 * mean_dist
    for i in range(n):
        j = local_opt_tour[i]
        penalized_dist[i, j] += tour_penalty_weight
        penalized_dist[j, i] += tour_penalty_weight
    
    # Ensure the matrix remains symmetric
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0
    
    # Ensure finite values
    penalized_dist = np.where(np.isfinite(penalized_dist), penalized_dist, edge_distance)
    
    # Check deadline - if we're running out of time, return immediately
    if time.time() - t_start > deadline - 5.0:
        return penalized_dist
    
    return penalized_dist
# EVOLVE-BLOCK-END
