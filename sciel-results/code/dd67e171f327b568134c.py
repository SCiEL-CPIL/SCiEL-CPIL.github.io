"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP using edge removal and targeted penalties.
    
    Strategy:
    1. Remove (set to large value) edges that have been used many times
    2. Apply strong penalty to current tour edges to force escape
    3. Keep the matrix finite and symmetric
    """
    n = edge_distance.shape[0]
    tour_len = len(local_opt_tour)
    
    # Start with a copy of the true distances
    perturbed = edge_distance.copy()
    
    # Identify edges in the current local optimum tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(tour_len):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % tour_len]
        tour_edges[u, v] = True
        tour_edges[v, u] = True
    
    # Strategy 1: Remove heavily used edges (set to a large but finite value)
    # This forces the search to avoid these edges entirely
    max_used = edge_n_used.max()
    if max_used > 0:
        # Edges used more than 50% of the max get removed
        removal_threshold = 0.5 * max_used
        to_remove = edge_n_used >= removal_threshold
        # Don't remove diagonal
        np.fill_diagonal(to_remove, False)
        # Set removed edges to a large value (10x the max distance)
        max_dist = edge_distance.max()
        removal_value = 10.0 * max_dist
        perturbed[to_remove] = removal_value
        perturbed[to_remove.T] = removal_value  # Ensure symmetry
    
    # Strategy 2: Strong penalty on current tour edges
    # This is the key perturbation - make current tour edges very expensive
    # so 2-opt must find a completely different structure
    tour_penalty = 5.0 * edge_distance.max()
    perturbed[tour_edges] += tour_penalty
    perturbed[tour_edges.T] += tour_penalty  # Ensure symmetry
    
    # Strategy 3: Moderate penalty on moderately used edges
    # Encourage diversity without completely removing options
    if max_used > 0:
        moderate_usage = (edge_n_used > 0) & (edge_n_used < removal_threshold)
        # Scale penalty by usage frequency
        usage_scale = edge_n_used[moderate_usage] / max_used
        penalty_values = 2.0 * edge_distance.max() * usage_scale
        # Apply only to moderate usage edges
        rows, cols = np.where(moderate_usage)
        perturbed[rows, cols] += penalty_values
        perturbed[cols, rows] += penalty_values  # Ensure symmetry
    
    # Ensure diagonal remains zero
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure all values are finite
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
