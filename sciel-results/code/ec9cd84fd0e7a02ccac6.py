import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturb the distance matrix by penalizing edges in the current local optimum tour.
    The penalty is proportional to the edge's usage history.
    """
    n = edge_distance.shape[0]
    # Create a copy to avoid modifying the original
    perturbed_distance = edge_distance.copy()
    
    # Get the edges in the current tour
    # local_opt_tour is a sequence of node indices, e.g., [0, 1, 2, ..., n-1, 0] or [0, 1, ..., n-1]
    # We assume it's a permutation of 0..n-1 representing the order.
    # The edges are (tour[0], tour[1]), (tour[1], tour[2]), ..., (tour[n-1], tour[0])
    
    if len(local_opt_tour) < 3:
        return perturbed_distance
        
    tour = np.array(local_opt_tour)
    
    # Calculate average edge length of the current tour to scale the penalty
    edges = [(tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))]
    
    # Compute lengths of current tour edges
    lengths = np.array([edge_distance[u, v] for u, v in edges])
    avg_len = np.mean(lengths) if len(lengths) > 0 else 1.0
    
    # If avg_len is 0, set to 1 to avoid division by zero or zero penalties
    if avg_len < 1e-9:
        avg_len = 1.0

    # Apply penalties
    # Base penalty factor: 0.5 means the edge cost increases by 50% of average edge length
    # Usage factor: adds 0.1 * avg_len per use
    for u, v in edges:
        usage = edge_n_used[u, v]
        # Penalty is additive
        penalty = avg_len * (0.5 + 0.1 * usage)
        
        # Update the distance matrix (symmetric)
        perturbed_distance[u, v] += penalty
        perturbed_distance[v, u] += penalty
        
    return perturbed_distance
