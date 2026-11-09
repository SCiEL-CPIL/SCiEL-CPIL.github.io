"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for the ACO solver.
    
    Uses a sharpened inverse-distance heuristic (1/d^2) to strongly bias
    the ants towards selecting the nearest unvisited customer, which
    typically leads to better initial route construction in CVRP instances
    of this size.
    """
    # Add a small epsilon to avoid division by zero
    d = distance_matrix + 1e-9
    # Squared inverse distance
    # This increases the contrast between short and long edges compared to 1/d
    eta = 1.0 / (d ** 2)
    
    # Ensure all values are positive and finite
    # The ACO code handles non-positive values, but we ensure our output is clean
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
