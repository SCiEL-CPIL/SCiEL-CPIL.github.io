"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Combines inverse distance with a demand-aware penalty to bias ants toward
    packing heavier customers, reducing the number of trips.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        NxN matrix of distances between nodes.
    coordinates : np.ndarray
        Nx2 array of coordinates.
    demands : np.ndarray
        Array of demands for each node.
    capacity : float
        Vehicle capacity.
    
    Returns:
    --------
    np.ndarray
        NxN heuristic matrix where larger values are more attractive.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=float)
    
    # Ensure demands is 1D
    if demands.ndim > 1:
        demands = demands.flatten()
    
    # Create a demand penalty: higher demand -> lower attractiveness
    # Use 1 / (1 + demand[j]) as a soft penalty
    demand_penalty = 1.0 / (1.0 + demands)
    
    # Inverse distance with a small epsilon to avoid division by zero
    # Use exponent 2 to make distance differences more pronounced
    inv_dist = 1.0 / (distance_matrix + 1e-9) ** 2
    
    # Combine: eta[i,j] = inv_dist[i,j] * demand_penalty[j]
    # This biases toward short edges to high-demand nodes
    eta = inv_dist * demand_penalty[np.newaxis, :]
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # The diagonal (self-loops) should be 0 or very small since we don't visit same node
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
