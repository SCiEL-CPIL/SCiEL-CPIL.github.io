"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search using adaptive z-score based additive penalty.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    n = edge_distance.shape[0]
    if n <= 1:
        return edge_distance.copy()
    
    # Get the edges in the current tour
    tour = local_opt_tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = (i + 1) % n
        tour_edges[tour[i], tour[j]] = True
        tour_edges[tour[j], tour[i]] = True
    
    # Compute statistics of edge usage for edges in the tour
    tour_usage = edge_n_used[tour_edges]
    if len(tour_usage) > 0:
        mean_usage = np.mean(tour_usage)
        std_usage = np.std(tour_usage)
        if std_usage < 1e-10:
            std_usage = 1.0
    else:
        mean_usage = 0.0
        std_usage = 1.0
    
    # Compute scale based on median distance
    lower_tri = np.tril_indices(n, k=-1)
    distances = edge_distance[lower_tri]
    non_zero_dist = distances[distances > 0]
    if len(non_zero_dist) > 0:
        scale = np.median(non_zero_dist)
    else:
        scale = 1.0
    
    # Adaptive alpha: scale penalty relative to distance scale
    # We want the penalty to be meaningful relative to typical edge lengths
    alpha = 0.3 * scale
    
    # Compute z-scores for edge usage
    # Only penalize edges with above-average usage
    z_scores = (edge_n_used - mean_usage) / std_usage
    # Clip to avoid extreme penalties
    z_scores = np.clip(z_scores, 0, 5)
    
    # Apply additive penalty based on z-scores
    perturbed_distance = edge_distance + alpha * z_scores
    
    # Ensure symmetry
    perturbed_distance = (perturbed_distance + perturbed_distance.T) / 2.0
    
    # Ensure no negative or infinite values
    perturbed_distance = np.nan_to_num(perturbed_distance, nan=0.0, posinf=1e6, neginf=0.0)
    perturbed_distance = np.maximum(perturbed_distance, 0.0)
    
    return perturbed_distance
# EVOLVE-BLOCK-END
