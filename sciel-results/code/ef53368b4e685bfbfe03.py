"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the heuristic matrix for Ant Colony Optimization.
    
    Uses inverse squared distance to create a steeper gradient, forcing ants
    to prefer nearest neighbors and form tight local clusters. This was
    measured to improve scores significantly in previous runs.
    """
    # Convert to float to ensure precision
    d = distance_matrix.astype(float)
    
    # Small epsilon to avoid division by zero (diagonal elements)
    eps = 1e-9
    
    # Inverse squared distance
    # This creates a sharper preference for close nodes compared to 1/d
    eta = 1.0 / (d + eps) ** 2
    
    # Ensure all values are positive and finite
    # Task requirement: "Heuristic entries that are not positive become 1e-9"
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
