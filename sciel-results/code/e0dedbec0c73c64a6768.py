"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Radial Basis Function (RBF) Approximation
    # This is a distinct approach from rational functions and Fourier series.
    # It uses a linear combination of Gaussian kernels centered at fixed points
    # in the input space. This allows for flexible local approximations.
    
    p = params
    
    # Define fixed centers in [0.5, 2]x[0.5, 2]x[0.5, 2]
    # We use 10 centers to match the 10 parameters.
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5]
    ])
    
    # Fixed width for all Gaussians
    sigma = 0.8
    
    # Compute RBF values
    # x, y, z are arrays of shape (N,)
    # centers are (10, 3)
    # We need to compute distance from each point (x_i, y_i, z_i) to each center
    
    # Expand x, y, z to (N, 1) to broadcast with centers (10, 3)
    # Actually, it's easier to loop or use broadcasting carefully.
    # Let's construct the input matrix X of shape (N, 3)
    X = np.column_stack((x, y, z))
    
    # Compute squared distances: ||X - C||^2
    # X: (N, 3), C: (10, 3)
    # dist_sq: (N, 10)
    # dist_sq[i, j] = (X[i] - C[j])^2
    
    # Using broadcasting:
    # X[:, np.newaxis, :] is (N, 1, 3)
    # C[np.newaxis, :, :] is (1, 10, 3)
    # diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]  (N, 10, 3)
    # dist_sq = np.sum(diff**2, axis=2)  (N, 10)
    
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Compute Gaussian values: exp(-dist_sq / (2 * sigma^2))
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # Linear combination: sum(p[j] * rbf_vals[i, j])
    # p is (10,), rbf_vals is (N, 10)
    # result is (N,)
    return np.sum(rbf_vals * p, axis=1)
# EVOLVE-BLOCK-END
