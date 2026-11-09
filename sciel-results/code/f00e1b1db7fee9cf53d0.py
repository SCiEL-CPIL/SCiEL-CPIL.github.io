import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Calculate a scale for the penalty based on the average edge length
    mean_dist = np.mean(edge_distance)
    
    # Base penalty proportional to usage count
    # Using a small multiplier to avoid overwhelming the true distances too early
    penalty_weight = 0.15 * mean_dist
    
    # Create a copy of the distance matrix to modify
    penalized_dist = edge_distance.copy()
    
    # Add penalty based on historical usage
    penalized_dist += penalty_weight * edge_n_used
    
    # Additionally, apply a strong penalty to the edges in the current local optimum
    # This ensures the next 2-opt step will likely modify these edges
    # We need to construct a mask for the edges in local_opt_tour
    n = len(local_opt_tour)
    # local_opt_tour is a list of indices [v0, v1, ..., v_{n-1}]
    # Edges are (v0, v1), (v1, v2), ..., (v_{n-2}, v_{n-1}), (v_{n-1}, v0)
    
    # We can vectorize the addition of the extra penalty
    # Extra penalty should be significant, e.g., 20% of mean distance
    extra_penalty = 0.2 * mean_dist
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        penalized_dist[u, v] += extra_penalty
        penalized_dist[v, u] += extra_penalty
        
    return penalized_dist
