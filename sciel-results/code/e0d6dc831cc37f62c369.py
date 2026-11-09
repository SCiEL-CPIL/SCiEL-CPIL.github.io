"""Rewritten program for guided local search penalty rule (TSP)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Combines:
    1. Global soft penalty proportional to edge usage, which gently discourages
       frequently-used edges across the whole matrix.
    2. Stronger targeted penalty on edges in the current local-optimal tour,
       scaled by their usage count, to specifically break the current solution.
    
    This dual approach ensures the search can escape local minima by making
    the current tour's edges expensive while also nudging away from edges
    that have been repeatedly used in previous iterations.
    """
    n = edge_distance.shape[0]
    # Make a copy so we don't mutate the input
    penalized = edge_distance.copy()
    
    # Extract edges from the current tour
    tour = np.asarray(local_opt_tour)
    m = len(tour)
    
    if m < 2:
        return penalized
    
    # Get max usage for normalization
    max_used = edge_n_used.max()
    if max_used <= 0:
        max_used = 1.0
    
    # Global soft penalty: apply a small penalty to all edges based on usage
    # This helps avoid cycles in the search by gently discouraging overused edges
    # Use a mild multiplicative factor
    global_penalty = 1.0 + 0.05 * (edge_n_used / max_used)
    penalized *= global_penalty
    
    # Targeted penalty on current tour edges
    # These edges are the ones we most want to break
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        used = edge_n_used[u, v]
        # Stronger penalty for tour edges, scaled by usage
        # Base penalty of 1.5x, plus additional scaling with usage
        # Higher usage means this edge has been stuck in tours longer, so penalize more
        tour_penalty = 1.5 + 0.3 * (used / max_used) + 0.05 * used
        # Cap to avoid extreme values
        tour_penalty = min(tour_penalty, 8.0)
        penalized[u, v] *= tour_penalty
        penalized[v, u] *= tour_penalty
    
    # Safety: ensure all values are finite
    if not np.all(np.isfinite(penalized)):
        penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
    
    return penalized
# EVOLVE-BLOCK-END
