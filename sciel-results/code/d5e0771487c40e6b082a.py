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
    
    # If no edges have been used yet, return original
    if edge_n_used.max() == 0:
        return edge_distance.copy()
    
    # Extract tour edges as a boolean mask
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_edges[i, j] = True
        tour_edges[j, i] = True
    
    # Compute average edge length of the current tour for scale awareness
    tour_len = 0.0
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[i, k]
    avg_edge_len = tour_len / n if n > 0 else 1.0
    
    # Use un-normalized usage counts with tuned additive penalty
    # This allows penalty magnitude to grow with iteration count,
    # increasing pressure to escape local optima in later rounds
    
    # Strong penalty for current tour edges (break local optimum)
    # Weaker penalty for other frequently-used edges (avoid oscillation)
    # Using global mean distance for more stable scaling
    mask = ~np.eye(n, dtype=bool)
    mean_dist = np.mean(edge_distance[mask]) if n > 1 else avg_edge_len
    
    alpha_tour = 0.025 * mean_dist
    alpha_global = 0.005 * mean_dist
    
    # Build penalty matrix:
    # Tour edges get stronger coefficient, others get weaker
    # Penalty scales linearly with usage count (not normalized)
    # so pressure increases over the 20 rounds
    penalty = np.where(tour_edges, alpha_tour, alpha_global) * edge_n_used
    
    # Additive penalty on top of true distances
    penalized = edge_distance + penalty
    
    # Ensure finiteness and positivity
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    penalized = np.maximum(penalized, 1e-10)
    
    # Symmetrize to guard against any numerical drift
    penalized = (penalized + penalized.T) / 2.0
    
    return penalized
# EVOLVE-BLOCK-END
