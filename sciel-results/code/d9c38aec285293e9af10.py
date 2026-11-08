# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    
    # Compute the true length of the current local-opt tour
    tour_len = 0.0
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[a, b]
    
    # GLS penalty: alpha * n_used, with alpha chosen so that
    # the penalized tour length is slightly above the current tour length.
    # We want: tour_len + alpha * sum(n_used on tour edges) = tour_len * (1 + margin)
    # => alpha = tour_len * margin / sum(n_used on tour edges)
    
    tour_edge_sum = 0.0
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edge_sum += edge_n_used[a, b]
    
    margin = 0.15  # 15% above current tour length
    if tour_edge_sum > 0:
        alpha = (tour_len * margin) / tour_edge_sum
    else:
        alpha = 0.0
    
    # Clamp alpha to a reasonable range
    alpha = np.clip(alpha, 0.0, 0.5)
    
    # Apply penalty: D_perturbed = D_true + alpha * edge_n_used
    D_perturbed = edge_distance + alpha * edge_n_used
    
    # Add tiny random perturbation to break symmetry
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1e-6, size=(n, n))
    D_perturbed += noise
    
    # Ensure symmetry
    D_perturbed = (D_perturbed + D_perturbed.T) / 2.0
    
    # Zero the diagonal
    np.fill_diagonal(D_perturbed, 0.0)
    
    # Ensure non-negativity and finiteness
    D_perturbed = np.maximum(D_perturbed, 0.0)
    D_perturbed = np.where(np.isfinite(D_perturbed), D_perturbed, edge_distance)
    
    return D_perturbed
# EVOLVE-BLOCK-END
