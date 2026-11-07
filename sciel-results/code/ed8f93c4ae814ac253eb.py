import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization.
    
    The heuristic is based on the 'opportunity cost' of visiting a node j from i.
    The cost is approximated by the sum of the distance to the node (d_ij) and the 
    distance from the node to the depot (d_j0), representing the 'detour' length.
    We square this cost to heavily penalize moves that take the ant far from the depot,
    ensuring the tour remains compact and feasible within the max_len constraint.
    
    eta[i, j] = prizes[j] / (d_ij + d_j0)^2
    """
    # Ensure inputs are numpy arrays for vectorized operations
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # d_j0 is the distance from each node j to the depot (node 0)
    # Shape: (N, 1)
    d_j0 = distance_matrix[:, 0][:, np.newaxis]
    
    # d_ij is the distance from i to j
    # Shape: (N, N)
    d_ij = distance_matrix
    
    # Cost of moving i -> j and eventually returning from j
    # Shape: (N, N)
    # Element (i, j) is d_ij[i, j] + d_j0[j]
    cost = d_ij + d_j0
    
    # Heuristic: Prize divided by squared cost
    # Add small epsilon to avoid division by zero
    epsilon = 1e-9
    eta = prizes[np.newaxis, :] / (cost + epsilon) ** 2
    
    # Ensure the result is a valid float matrix
    return eta.astype(np.float64)
