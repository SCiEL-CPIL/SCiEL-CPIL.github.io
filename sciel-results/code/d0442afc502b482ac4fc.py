"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies a Guided Local Search penalty to the edges of the current local optimal tour.
    The penalty is additive and proportional to the number of times the edge has been used
    in previous local optima, scaled by the mean edge distance of the instance.
    """
    n = edge_distance.shape[0]
    # Create a copy of the distance matrix to avoid modifying the input
    perturbed = edge_distance.copy()
    
    # Calculate the mean edge distance from the upper triangle to scale the penalty
    # This ensures the penalty is relative to the magnitude of the distances
    iu = np.triu_indices(n, k=1)
    if len(iu[0]) > 0:
        mean_dist = np.mean(edge_distance[iu])
    else:
        mean_dist = 1.0
        
    # Penalty coefficient: 0.5 times the mean edge distance per use.
    # This makes the penalty significant (50% of average edge cost per use)
    alpha = 0.5 * mean_dist
    
    # Apply penalties to the edges present in the current local optimal tour
    m = len(local_opt_tour)
    for i in range(m):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % m]
        
        # Retrieve the usage count for this edge
        count = edge_n_used[u, v]
        
        if count > 0:
            penalty = alpha * count
            # Add the penalty to the symmetric entries
            perturbed[u, v] += penalty
            perturbed[v, u] += penalty
            
    return perturbed
# EVOLVE-BLOCK-END
