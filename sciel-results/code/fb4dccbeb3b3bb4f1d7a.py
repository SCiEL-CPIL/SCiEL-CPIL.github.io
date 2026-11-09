"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    # Start with a copy of the true distances
    perturbed = edge_distance.copy()
    
    # Calculate the average edge length of the current tour to scale the penalty
    # This adapts the penalty strength to the specific instance geometry
    tour_edges = local_opt_tour[:-1]
    tour_next = local_opt_tour[1:]
    
    # Ensure we handle the closing edge if the tour is represented as a cycle
    # Usually local_opt_tour is a permutation, so the tour is closed by connecting last to first
    # Get the true distances for edges in the current tour
    tour_dist = edge_distance[tour_edges, tour_next]
    
    # Also add the closing edge distance
    if n > 1:
        closing_dist = edge_distance[local_opt_tour[-1], local_opt_tour[0]]
        tour_dist = np.concatenate([tour_dist, [closing_dist]])
    
    avg_tour_edge = np.mean(tour_dist) if len(tour_dist) > 0 else 1.0
    
    # Global usage penalty: penalize all edges based on their historical usage
    # This creates a stronger repulsion from the entire history of local optima
    # Use a smooth scaling to avoid extreme values
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1.0
    if max_used > 0:
        # Normalize usage counts to [0, 1]
        normalized_usage = edge_n_used / (1.0 + max_used)
        
        # Penalty factor: edges that have been used more get penalized more
        # Use a power law to make the penalty more selective
        penalty_factor = 1.0 + 0.15 * np.power(normalized_usage, 1.5)
        
        # Apply the global penalty
        perturbed = perturbed * penalty_factor
    
    # Additional penalty for edges in the current local optimum tour
    # This helps escape the current basin more aggressively
    for i in range(n - 1):
        u = local_opt_tour[i]
        v = local_opt_tour[i + 1]
        # Add extra penalty for edges in the current tour
        perturbed[u, v] *= 1.0 + 0.2
        perturbed[v, u] *= 1.0 + 0.2
    
    # Add penalty for the closing edge
    if n > 1:
        u = local_opt_tour[-1]
        v = local_opt_tour[0]
        perturbed[u, v] *= 1.0 + 0.2
        perturbed[v, u] *= 1.0 + 0.2
    
    # Ensure the matrix is symmetric
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure finite values
    perturbed = np.nan_to_num(perturbed, nan=1.0, posinf=1e10, neginf=0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
