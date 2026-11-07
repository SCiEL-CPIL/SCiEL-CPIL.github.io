import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Safety margin
    if time.time() - start_time > 280:
        return edge_distance.copy()
    
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    tour = local_opt_tour
    if len(tour) != n:
        tour = list(range(n))
    
    # Calculate penalty
    # Use a stronger penalty to encourage exploration
    alpha = 1.5
    beta = 1.0
    
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used == 0:
        # Still add a small penalty to current tour edges to encourage change
        immediate_penalty = np.zeros_like(edge_n_used)
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            if u != v:
                immediate_penalty[u, v] += 0.5
                immediate_penalty[v, u] += 0.5
        d_out = edge_distance * (1.0 + immediate_penalty)
        d_out = (d_out + d_out.T) / 2.0
        np.fill_diagonal(d_out, 0.0)
        d_out[~np.isfinite(d_out)] = edge_distance.max()
        return d_out
        
    # Saturating penalty with stronger alpha
    penalty = alpha * (edge_n_used / (edge_n_used + beta))
    
    # Add extra penalty for edges in the current local optimum
    # Stronger immediate penalty to ensure we change the current tour
    immediate_penalty = np.zeros_like(edge_n_used)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if u != v:
            immediate_penalty[u, v] += 1.0
            immediate_penalty[v, u] += 1.0
            
    total_penalty = penalty + immediate_penalty
    
    # Apply multiplicatively
    d_out = edge_distance * (1.0 + total_penalty)
    
    # Symmetrize and zero diagonal
    d_out = (d_out + d_out.T) / 2.0
    np.fill_diagonal(d_out, 0.0)
    
    # Ensure finite
    d_out[~np.isfinite(d_out)] = edge_distance.max()
    
    # Check time
    if time.time() - start_time > 285:
        return edge_distance.copy()
    
    return d_out
