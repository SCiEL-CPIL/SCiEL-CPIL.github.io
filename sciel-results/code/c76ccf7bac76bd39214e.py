import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    start_time = time.time()
    n = edge_distance.shape[0]
    # Copy the matrix to avoid side effects
    perturbed = edge_distance.copy()
    
    # Get the edges of the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))
        
    # Calculate mean edge length in the current tour
    if len(tour_edges) > 0:
        tour_lengths = [edge_distance[u, v] for u, v in tour_edges]
        mean_edge_len = np.mean(tour_lengths)
    else:
        mean_edge_len = 1.0
        
    # Determine the maximum usage count among the edges in the current tour
    max_count = 0
    for u, v in tour_edges:
        c = edge_n_used[u, v]
        if c > max_count:
            max_count = c
            
    # Penalty parameter base
    # Using a slightly lower base scale to balance with the normalized quadratic term
    penalty_scale = 0.12 * mean_edge_len
    
    # Apply normalized quadratic penalty to edges in the current tour
    for u, v in tour_edges:
        count = edge_n_used[u, v]
        if count > 0:
            if max_count > 0:
                norm_count = count / max_count
            else:
                norm_count = 0.0
            # Quadratic scaling of the normalized count
            penalty = penalty_scale * (norm_count ** 2)
            perturbed[u, v] += penalty
            perturbed[v, u] += penalty
            
    # Time guard: if we're close to the deadline, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return perturbed
        
    return perturbed
