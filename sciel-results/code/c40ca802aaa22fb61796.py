"""Guided local search penalty rule for TSP - geometry-aware repulsive force model."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Apply a geometry-aware repulsive penalty to the distance matrix.
    
    The penalty is inversely proportional to edge distance (so short edges get
    disproportionately large penalties) and scaled by usage frequency. This
    forces the search to explore alternative short connections and escape
    the current local optimum basin.
    
    Args:
        edge_distance: true Euclidean distance matrix (n x n)
        local_opt_tour: current tour as a sequence of node indices
        edge_n_used: symmetric count matrix of edge usage
    
    Returns:
        Perturbed distance matrix with penalties applied.
    """
    n = edge_distance.shape[0]
    
    # Identify edges in the current tour
    tour_edges = np.zeros((n, n), dtype=np.float64)
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0
    
    # Normalize usage counts
    max_used = edge_n_used.max()
    if max_used > 0:
        normalized_usage = edge_n_used / max_used
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Compute inverse-distance weight: short edges get higher weights
    # Use a smoothed version to avoid division by zero and extreme values
    eps = 1e-8
    inv_dist = 1.0 / (edge_distance + eps)
    
    # Scale inverse distance to a reasonable range
    # Normalize so that the mean inverse distance is 1
    mean_inv = inv_dist[tour_edges > 0].mean() if np.any(tour_edges > 0) else 1.0
    if mean_inv > 0:
        inv_dist_normalized = inv_dist / mean_inv
    else:
        inv_dist_normalized = inv_dist
    
    # Combine usage frequency with geometry-aware weight
    # Penalty is proportional to usage * inverse_distance
    # This makes short, frequently-used edges very costly
    penalty_weight = normalized_usage * inv_dist_normalized
    
    # Apply a moderate penalty coefficient
    # The penalty should be strong enough to escape local optima
    # but not so strong that it destroys the distance structure
    alpha = 0.3
    
    # Only apply penalty to edges that are either in the current tour
    # or have been used before
    mask = (tour_edges > 0) | (edge_n_used > 0)
    
    # Compute perturbed distance
    perturbed = edge_distance.copy()
    perturbed[mask] += alpha * penalty_weight[mask] * edge_distance[mask]
    
    # Ensure the matrix remains finite and symmetric
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(perturbed, 0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
