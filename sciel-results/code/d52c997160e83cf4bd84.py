# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Topological disruption strategy for guided local search.
    
    Instead of scalar penalties, we identify the k shortest edges
    in the current tour and set their distances to a large value, 
    forcing 2-opt to find alternative connections and restructure the tour.
    """
    n = edge_distance.shape[0]
    
    # Work on a copy to avoid modifying the input
    perturbed = edge_distance.copy()
    
    # Extract the edges of the current tour
    # local_opt_tour is a permutation of [0, ..., n-1] representing the tour
    # The tour edges are (tour[i], tour[i+1]) for i in range(n), with wrap-around
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Ensure consistent ordering for the symmetric matrix
        if u > v:
            u, v = v, u
        tour_edges.append((u, v))
        
    # Get the true distances for the tour edges
    # We want to identify the k shortest edges in the current tour
    tour_edge_data = []
    for u, v in tour_edges:
        dist = edge_distance[u, v]
        tour_edge_data.append((dist, u, v))
        
    # Sort by distance in ascending order (shortest first)
    tour_edge_data.sort(key=lambda x: x[0])
    
    # Determine how many edges to disrupt
    # A small number of edges (e.g., 3) is typically sufficient for perturbation
    # without completely destroying the tour structure
    k_disrupt = 3
    
    # Set the k shortest edges to a large value to effectively "remove" them
    large_value = 1000.0 # Much larger than max possible distance in unit square (~1.414)
    
    for i in range(k_disrupt):
        dist, u, v = tour_edge_data[i]
        perturbed[u, v] = large_value
        perturbed[v, u] = large_value
    
    # Ensure the matrix is symmetric
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure no NaN or Inf values
    perturbed = np.nan_to_num(perturbed, nan=large_value, posinf=large_value, neginf=0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
