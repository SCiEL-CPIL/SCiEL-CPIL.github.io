import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Updates the edge distance matrix by adding penalties to edges
    frequently used in previous local optima (Guided Local Search).
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin, though N=100 is fast
    
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    # Extract edges of the current local optimum tour
    tour = np.asarray(local_opt_tour)
    
    # Calculate the average edge length of the current tour using true distances
    u = tour
    v = np.roll(tour, -1)
    
    # Get true distances for these edges
    current_edge_lengths = edge_distance[u, v]
    avg_edge_len = np.mean(current_edge_lengths)
    
    # If the tour is degenerate or something weird, fallback to global mean or small constant
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = np.mean(edge_distance)
        if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
            avg_edge_len = 0.1 # Fallback for unit square 100 pts

    # Penalty coefficients
    # alpha: scale for accumulated usage penalty
    # beta: scale for current tour penalty (diversification boost)
    alpha = 0.1
    beta = 0.5
    
    # Base penalty from accumulated usage
    penalized_dist = edge_distance + (alpha * avg_edge_len) * edge_n_used
    
    # Add extra penalty to edges in the current tour to force diversification
    # This is the core of Guided Local Search: penalize the current local optimum
    penalized_dist[u, v] += beta * avg_edge_len
    penalized_dist[v, u] += beta * avg_edge_len
    
    # Ensure the diagonal is 0 (or very small) to avoid self-loops if 2-opt implementation is naive
    np.fill_diagonal(penalized_dist, 0.0)
    
    # Ensure symmetry (just in case, though inputs should be symmetric)
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0
    
    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=np.max(edge_distance), neginf=0.0)
        
    # Time check (unlikely to trigger for N=100)
    if time.time() > deadline:
         return edge_distance.copy()

    return penalized_dist
