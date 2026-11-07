"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP using Rank-Based Adaptive Penalties.
    
    Args:
        edge_distance: true Euclidean distance matrix (n x n)
        local_opt_tour: current tour as a sequence of node indices
        edge_n_used: symmetric count matrix of how often each edge has been used
    
    Returns:
        Perturbed distance matrix with rank-based penalties applied.
    """
    n = edge_distance.shape[0]
    
    max_used = edge_n_used.max()
    if max_used == 0:
        return edge_distance.copy()
        
    # Rank-based penalty: 
    # Higher usage count -> Higher rank -> Higher penalty.
    # This avoids issues with a single outlier edge dominating the penalty scale.
    flat_usage = edge_n_used.flatten()
    # Use argsort of argsort to get ranks (0 to n*n-1)
    # We want higher usage to have higher rank.
    ranks = np.argsort(np.argsort(flat_usage)).reshape(n, n)
    
    # Normalize ranks to [0, 1]
    max_rank = ranks.max()
    if max_rank > 0:
        normalized_ranks = ranks / max_rank
    else:
        normalized_ranks = np.zeros_like(ranks)
        
    # Adaptive strength: increases as the search progresses (max_used increases)
    # Start with a moderate penalty, increase it over time.
    # max_used is roughly the number of iterations.
    base_strength = 0.2
    adaptive_factor = 1.0 + 0.05 * max_used
    strength = base_strength * adaptive_factor
    
    # Calculate penalty matrix
    # We only want to penalize off-diagonal edges
    penalty_matrix = strength * normalized_ranks
    np.fill_diagonal(penalty_matrix, 0.0)
    
    # Additional penalty for edges in the current local optimum tour
    # This forces the 2-opt to consider breaking these edges.
    tour_edges = np.zeros((n, n), dtype=bool)
    tour_len = len(local_opt_tour)
    for i in range(tour_len):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % tour_len]
        tour_edges[u, v] = True
        tour_edges[v, u] = True
        
    # Stronger penalty for current tour edges to ensure escape
    tour_penalty_strength = strength * 1.5
    penalty_matrix[tour_edges] += tour_penalty_strength
    
    # Apply penalty to the distance matrix
    perturbed = edge_distance + penalty_matrix
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure finite values
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
