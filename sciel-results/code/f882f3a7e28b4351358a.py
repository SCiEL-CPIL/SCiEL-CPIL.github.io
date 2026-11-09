"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF model with 10 fixed centers evenly spaced across [-3, 3]
    # Centers: -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4
    centers = np.linspace(-3.0, 2.4, 10)
    
    # Width parameter - moderate value for smooth basis functions
    # Using sigma = 0.8 for a balance between locality and coverage
    sigma = 0.8
    
    # Compute RBF basis: exp(-((x - c)^2) / (2 * sigma^2))
    # x has shape (N,), centers has shape (10,)
    # We want to compute for each x_i and each center c_j: exp(-((x_i - c_j)^2) / (2 * sigma^2))
    
    # Reshape for broadcasting: x[:, np.newaxis] - centers[np.newaxis, :]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # rbf_values has shape (N, 10)
    # params has shape (10,)
    # Result = rbf_values @ params
    
    result = rbf_values @ params
    
    return result
# EVOLVE-BLOCK-END
