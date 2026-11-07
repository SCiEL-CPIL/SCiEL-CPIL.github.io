"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed RBF centers: 3x3 grid plus origin (10 total)
    centers = np.array([
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])
    
    # Fixed bandwidth for RBFs (Gaussian kernels)
    sigma = 0.5
    sigma2 = 2.0 * sigma * sigma
    
    # x, y are 1D arrays of shape (N,)
    x_col = x[:, np.newaxis]  # (N, 1)
    y_col = y[:, np.newaxis]  # (N, 1)
    
    cx = centers[:, 0]  # (10,)
    cy = centers[:, 1]  # (10,)
    
    # Compute squared distances: (N, 10)
    dx = x_col - cx[np.newaxis, :]  # (N, 10)
    dy = y_col - cy[np.newaxis, :]  # (N, 10)
    dist2 = dx * dx + dy * dy  # (N, 10)
    
    # Gaussian RBFs: exp(-dist2 / sigma2)
    rbf = np.exp(-dist2 / sigma2)  # (N, 10)
    
    # params has length 10
    # Ensure params is 1D
    p = params.reshape(10)
    
    # Linear combination
    prediction = rbf @ p
    
    return prediction
# EVOLVE-BLOCK-END
