"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Extract tour edges
    tour_edges = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_edges[i, j] = 1.0
        tour_edges[j, i] = 1.0
    
    # Compute average edge length of the current tour for scale awareness
    tour_len = 0.0
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[i, k]
    avg_edge_len = tour_len / n if n > 0 else 1.0
    
    # Normalize usage counts
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used <= 0:
        return edge_distance.copy()
    
    normalized_usage = edge_n_used / max_used
    
    # Adaptive penalty: scale by average edge length so penalty is meaningful
    # relative to the problem's distance scale
    # Stronger penalty for edges in the current tour (to break local optimum)
    # Weaker but still present penalty for all frequently-used edges (to avoid oscillation)
    
    # Penalty for tour edges: strong, scaled by usage and average edge length
    alpha_tour = 0.8 * avg_edge_len
    # Penalty for non-tour edges: weaker, global usage penalty
    alpha_global = 0.1 * avg_edge_len
    
    # Build penalty matrix
    # For tour edges: use stronger coefficient
    # For non-tour edges: use weaker coefficient
    penalty = np.where(tour_edges > 0, alpha_tour, alpha_global) * normalized_usage
    
    # Additive penalty on top of true distances
    penalized = edge_distance + penalty
    
    # Ensure finiteness and positivity
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    penalized = np.maximum(penalized, 1e-10)
    
    # Symmetrize to guard against any numerical drift
    penalized = (penalized + penalized.T) / 2.0
    
    return penalized
# EVOLVE-BLOCK-END
