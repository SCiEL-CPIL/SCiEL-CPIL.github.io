import numpy as np

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
    # Convert to numpy arrays if not already
    edge_distance = np.asarray(edge_distance, dtype=float)
    edge_n_used = np.asarray(edge_n_used, dtype=float)
    local_opt_tour = np.asarray(local_opt_tour, dtype=int)
    
    # Normalize usage counts to prevent extreme penalties
    max_used = edge_n_used.max()
    if max_used > 0:
        norm_used = edge_n_used / (1.0 + max_used)
    else:
        norm_used = np.zeros_like(edge_n_used)
        
    # Copy the true distances
    result = edge_distance.copy()
    
    # 1. Global penalty: Slightly increase the cost of edges that have been used frequently
    # This helps in escaping basins of attraction that have been visited before.
    result *= (1.0 + 0.2 * norm_used)
    
    # 2. Local penalty: Strongly penalize the edges of the current local optimum tour
    # This is the core of GLS: forcing the search to break the current solution.
    n = len(local_opt_tour)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Penalty is proportional to how much this edge has been used in the past
        # Base penalty of 0.1, scaled by usage.
        # For N=100, typical edge length is ~0.01-0.05. 0.1 is a strong penalty.
        penalty = 0.1 * (1.0 + norm_used[u, v])
        
        result[u, v] += penalty
        result[v, u] += penalty
        
    return result
