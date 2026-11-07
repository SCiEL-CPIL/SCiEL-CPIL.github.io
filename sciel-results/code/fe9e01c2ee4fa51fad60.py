import numpy as np
import time

_start_time = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _start_time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP.
    
    Applies focused penalties to current tour edges with moderate strength,
    scaled relative to the problem's distance distribution.
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Compute scale factor: median of non-diagonal distances
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (edge_distance > 1e-10) & mask
    if np.any(nonzero_mask):
        dist_scale = np.median(edge_distance[nonzero_mask])
    else:
        dist_scale = 1.0
    
    # Ensure dist_scale is positive and reasonable
    if dist_scale <= 0 or not np.isfinite(dist_scale):
        dist_scale = 1.0
    
    # Start with a copy of the true distances
    result = edge_distance.astype(np.float64, copy=True)
    
    # Extract current tour edges
    tour = local_opt_tour
    if len(tour) < 2:
        return result
    
    # Build the set of tour edges (i, j) where i < j for symmetry
    tour_edges = set()
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        tour_edges.add((a, b))
    
    # Apply penalty only to current tour edges
    # Use a multiplicative penalty that scales with usage count
    # Moderate strength to encourage restructuring without over-distorting
    for (i, j) in tour_edges:
        usage = edge_n_used[i, j]
        # Penalty factor: 1 + alpha * log1p(usage)
        # alpha = 0.25 gives moderate penalty
        penalty_factor = 1.0 + 0.25 * np.log1p(usage)
        result[i, j] *= penalty_factor
        result[j, i] *= penalty_factor
    
    # Ensure finiteness and non-negativity
    result = np.where(np.isfinite(result), result, edge_distance)
    result = np.maximum(result, 1e-10)
    
    return result
