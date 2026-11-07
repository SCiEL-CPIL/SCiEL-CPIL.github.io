"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a Nearest Neighbor Heuristic.
    
    This approach identifies the nearest neighbor for each node and boosts the
    attractiveness of these specific edges. This encourages ants to follow
    locally optimal paths, which often form the core of the global optimal tour.
    """
    n = distance_matrix.shape[0]
    
    # Base: inverse distance
    # Avoid division by zero for diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = 1.0 / distance_matrix
    inv_dist[~np.isfinite(inv_dist)] = 0.0
    
    # Initialize boost matrix with 1.0 (no change)
    boost = np.ones((n, n))
    
    # Find 1-Nearest Neighbor for each node
    # We need to ignore the self-loop (distance 0)
    # Create a copy of distance matrix with diagonal set to inf
    dist_for_nn = distance_matrix.copy()
    np.fill_diagonal(dist_for_nn, np.inf)
    
    # For each node i, find the index j with minimum distance
    nn_indices = np.argmin(dist_for_nn, axis=1)
    
    # Boost the edge (i, nn[i]) and (nn[i], i)
    # This is a directed boost, but since the tour can go either way, 
    # and the heuristic is used for transition probability, 
    # boosting both directions helps.
    for i in range(n):
        j = nn_indices[i]
        # Boost factor: significantly higher than base
        # If we use 3.0, edges to NN are 3x more attractive than a same-distance non-NN edge
        boost[i, j] = 3.0
        boost[j, i] = 3.0
        
    # Combine base and boost
    heuristic = inv_dist * boost
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
# EVOLVE-BLOCK-END
