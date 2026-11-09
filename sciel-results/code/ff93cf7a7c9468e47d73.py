"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t_start = time.time()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Fixed-width RBF model with 10 centers evenly spaced across [-3, 3]
    # Centers: -3, -2.333, -1.667, -1, -0.333, 0.333, 1, 1.667, 2.333, 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Moderate global width for smooth interpolation
    width = 0.8
    
    # Build RBF basis: phi_i(x) = exp(-((x - c_i)^2) / (2 * width^2))
    # Shape: (10, n_points)
    diff = x[None, :] - centers[:, None]  # (10, n_points)
    basis = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    if time.time() - t_start > 100:
        return np.zeros(n_points)
    
    return np.einsum('i,ij->j', params, basis)
# EVOLVE-BLOCK-END
