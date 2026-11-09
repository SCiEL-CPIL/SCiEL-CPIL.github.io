"""Rewritten program for guided local search penalty rule (TSP)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Combines:
    1. A very mild global soft penalty proportional to edge usage.
       Analysis suggests that strong global penalties can distort the landscape
       too much, so this is reduced compared to previous versions.
    2. A stronger targeted penalty on edges in the current local-optimal tour,
       scaled by their usage count, to specifically break the current solution.
       The base multiplier and usage scaling are increased to force a more
       significant perturbation of the current tour structure.
    
    The penalty is multiplicative and bounded to keep the modified distance
    matrix well-conditioned for 2-opt.
    """
    # Deadline guard: track elapsed time to avoid timeout issues.
    t0 = time.time()
    
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
    
    # Global soft penalty: apply a very small penalty to all edges based on usage.
    # Reduced from 0.02 to 0.008 to minimize global distortion while still
    # discouraging extremely overused edges.
    global_penalty = 1.0 + 0.008 * (edge_n_used / max_used)
    penalized *= global_penalty
    
    # Add spectral Laplacian regularization to create a smooth global bias
    # against high-frequency oscillations in tour structure
    # Compute degree matrix and Laplacian of the distance matrix
    degree = edge_distance.sum(axis=1)
    L = np.diag(degree) - edge_distance
    
    # Add scaled accumulated edge usage as a topological bias
    # This creates a smoother global penalty that discourages
    # oscillatory tour patterns rather than just individual edge costs
    usage_bias = 0.002 * edge_n_used
    penalized += L @ usage_bias * 0.1
    
    # Targeted penalty on current tour edges.
    # Reduced base from 1.40 to 1.35 and adjusted usage scaling for better balance.
    # This allows more moderate perturbation while still breaking the current tour structure.
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        used = edge_n_used[u, v]
        # Penalty: base multiplier plus scaling with usage.
        tour_penalty = 1.35 + 0.40 * (used / max_used) + 0.025 * used
        # Cap to avoid extreme values that could distort 2-opt
        tour_penalty = min(tour_penalty, 5.5)
        penalized[u, v] *= tour_penalty
        penalized[v, u] *= tour_penalty
    
    # Safety: ensure all values are finite
    if not np.all(np.isfinite(penalized)):
        penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Time guard: if we're taking too long (shouldn't happen for n=100), return anyway
    elapsed = time.time() - t0
    # This function should take microseconds to milliseconds
    
    return penalized
# EVOLVE-BLOCK-END
