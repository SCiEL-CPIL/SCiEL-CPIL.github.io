"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Combines:
    1. Saturating usage penalty (smooth, bounded)
    2. Geometry-aware repulsive force (inversely proportional to distance)
       so short, high-value edges get disproportionately large penalties
    3. Immediate penalty on current tour edges for guaranteed escape
    
    The multiplicative application keeps distances positive and well-behaved
    for 2-opt.
    """
    n = edge_distance.shape[0]
    
    # --- Usage-based penalty with saturating normalization ---
    # alpha controls max penalty factor; beta controls saturation speed
    alpha = 0.55
    beta = 1.5
    penalty = alpha * edge_n_used / (beta + edge_n_used)
    
    # --- Geometry-aware repulsive force ---
    # Short edges (small distance) get larger penalty to force exploration
    # of alternative short connections. Use inverse distance normalized to [0,1].
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = 1.0 / (edge_distance + 1e-8)
    
    # Zero out diagonal (self-loops don't matter)
    np.fill_diagonal(inv_dist, 0.0)
    
    # Normalize inverse distance to [0, 1] over off-diagonal entries
    off_diag_mask = ~np.eye(n, dtype=bool)
    min_inv = inv_dist[off_diag_mask].min()
    max_inv = inv_dist[off_diag_mask].max()
    if max_inv > min_inv:
        inv_dist_norm = (inv_dist - min_inv) / (max_inv - min_inv)
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    np.fill_diagonal(inv_dist_norm, 0.0)
    
    # Scale geometry penalty to be comparable to usage penalty
    gamma = 0.35
    penalty += gamma * inv_dist_norm
    
    # --- Immediate penalty on edges in the current local-opt tour ---
    # Guarantees perturbation away from the current basin
    immediate_scale = 0.15
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    penalty += immediate_scale * tour_mask
    
    # --- Apply penalty multiplicatively to keep distances positive ---
    perturbed = edge_distance * (1.0 + penalty)
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure all values are finite
    perturbed = np.where(np.isfinite(perturbed), perturbed, 0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
