"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Strategy:
    - Strongly penalize edges in the current local optimum tour to force escape
    - Use accumulated usage history for additional historical penalty
    - Apply exponential saturation for smooth, bounded penalty scaling
    - Only penalize edges that are actually in the current tour (not all used edges)
      This keeps non-tour distances undistorted, forcing 2-opt to restructure
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return edge_distance.copy()
    
    # Create a copy to work with
    result = edge_distance.copy()
    
    # Build the set of edges in the current tour
    tour_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Get usage counts for tour edges
    max_usage = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_usage == 0:
        max_usage = 1.0
    
    # Apply penalty to tour edges with graduated scaling
    # Lower base penalty to avoid over-perturbing; let historical usage drive strength
    for (u, v) in tour_edges:
        usage = edge_n_used[u, v]
        # Smooth saturation: penalty grows with usage but saturates
        # Using a logistic-like curve for smoothness
        normalized = usage / (1.0 + max_usage)
        # Soft penalty: starts moderate, increases with repeated usage
        base_penalty = 0.3
        hist_penalty = 0.4 * np.tanh(normalized)  # smooth, bounded [0, 0.4]
        total_penalty = 1.0 + base_penalty + hist_penalty
        result[u, v] *= total_penalty
        result[v, u] *= total_penalty
    
    # Light penalty on high-usage non-tour edges to encourage diversity
    # This helps break symmetry and explore alternative edges
    mask = np.ones((n, n), dtype=bool)
    for (u, v) in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    mask[np.diag_indices(n)] = False  # exclude self-loops
    
    # Find top-usage non-tour edges and apply a small penalty
    usage_masked = np.where(mask, edge_n_used, 0)
    max_non_tour_usage = usage_masked.max() if usage_masked.size > 0 else 0
    if max_non_tour_usage > 0:
        # Apply gentle penalty to edges with usage above 70% of max
        threshold = 0.7 * max_non_tour_usage
        high_usage_mask = (edge_n_used > threshold) & mask
        if np.any(high_usage_mask):
            for i in range(n):
                for j in range(i + 1, n):
                    if high_usage_mask[i, j]:
                        penalty = 1.0 + 0.15 * (edge_n_used[i, j] / max_non_tour_usage)
                        result[i, j] *= penalty
                        result[j, i] *= penalty
    
    return result
# EVOLVE-BLOCK-END
