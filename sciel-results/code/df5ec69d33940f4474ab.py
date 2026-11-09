import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Updates the edge distance matrix by adding penalties to edges in the current
    local optimal tour and edges that have been frequently used in previous iterations.
    This implements a guided local search perturbation strategy.
    """
    n = len(local_opt_tour)
    
    # Calculate the mean edge distance for calibration
    # Using the mean of the upper triangle to avoid double counting zeros on diagonal if any, 
    # but edge_distance is a full matrix. Mean of all non-zero or just mean?
    # Mean of the matrix is fine as a scale factor.
    mean_dist = np.mean(edge_distance)
    
    # Initialize the penalty matrix to zeros
    penalty = np.zeros_like(edge_distance)
    
    # 1. Penalize edges in the current local optimal tour
    # This encourages 2-opt to remove these specific edges.
    # A coefficient of 0.2 is a moderate nudge relative to the mean edge length.
    alpha_tour = 0.2 * mean_dist
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        penalty[u, v] += alpha_tour
        penalty[v, u] += alpha_tour
        
    # 2. Penalize edges based on historical usage
    # This prevents the search from cycling back to previously visited local optima.
    # Normalize the usage counts to be in the range [0, 1].
    max_used = edge_n_used.max()
    if max_used > 0:
        norm_usage = edge_n_used / (1.0 + max_used)
    else:
        norm_usage = np.zeros_like(edge_n_used)
        
    # A smaller coefficient for historical penalties to prioritize the current tour escape
    alpha_hist = 0.1 * mean_dist
    penalty += alpha_hist * norm_usage
    
    # Return the modified distance matrix
    return edge_distance + penalty
