# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Topological disruption strategy for guided local search.
    
    Identify the k shortest edges in the current tour and remove them
    from the feasible set by setting their distance to a large constant,
    forcing 2-opt to restructure the tour around alternative connections.
    """
    n = edge_distance.shape[0]
    
    # Work on a copy to avoid modifying the input
    perturbed = edge_distance.copy()
    
    # Extract the edges of the current tour with their distances
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        dist = edge_distance[u, v]
        tour_edges.append((dist, u, v))
    
    # Sort by distance in ascending order (shortest first)
    tour_edges.sort(key=lambda x: x[0])
    
    # Number of shortest edges to disrupt
    # Use a fraction of the tour edges - around 12% for n=100
    k_disrupt = max(1, int(0.12 * n))
    
    # Set the k shortest tour edges to a large value
    large_value = 1e6
    
    for i in range(k_disrupt):
        dist, u, v = tour_edges[i]
        perturbed[u, v] = large_value
        perturbed[v, u] = large_value
    
    # Ensure the matrix is symmetric
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure no NaN or Inf values
    perturbed = np.nan_to_num(perturbed, nan=large_value, posinf=large_value, neginf=0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
