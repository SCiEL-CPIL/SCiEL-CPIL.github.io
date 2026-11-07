"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) interpolant for E. coli growth rate.
    Uses 10 fixed Gaussian centers in the normalized input space.
    The 10 params are the weights for each basis function.
    """
    # Normalize inputs to [0, 1] range based on given bounds
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Stack normalized inputs into a matrix of shape (N, 4)
    X = np.column_stack((b_norm, s_norm, temp_norm, ph_norm))
    
    # Define 10 fixed centers in the normalized 4D space
    # 4 corners: (0,0,0,0), (1,1,1,1), (0,0,1,1), (1,1,0,0)
    # 4 mid-edges: (0.5,0,0,0), (0,0.5,0,0), (0,0,0.5,0), (0,0,0,0.5)
    # 1 center: (0.5, 0.5, 0.5, 0.5)
    # 1 extra: (0.5, 0.5, 0.5, 0.5) - wait, need 10 distinct centers.
    # Let's use:
    # 4 corners
    # 4 face centers of the hypercube? No, let's just pick 10 well-distributed points.
    
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 1.0, 0.5, 0.5],
        [1.0, 0.0, 0.5, 0.5],
        [0.5, 0.0, 1.0, 0.5],
        [0.5, 1.0, 0.0, 0.5],
        [0.5, 0.5, 0.0, 1.0]
    ])
    
    # Compute squared distances from each data point to each center
    # X has shape (N, 4), centers has shape (10, 4)
    # dist_sq[i, j] = ||X[i] - centers[j]||^2
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # RBF width (sigma). We can fix it or make it a parameter. 
    # Since we only have 10 params for weights, let's fix sigma.
    # A good default for unit hypercube is around 0.5-1.0.
    sigma = 0.8
    sigma_sq = sigma ** 2
    
    # Compute RBF values: exp(-dist_sq / (2 * sigma_sq))
    rbf_values = np.exp(-dist_sq / (2.0 * sigma_sq))
    
    # rbf_values has shape (N, 10)
    # params has shape (10,)
    # prediction = rbf_values @ params
    
    prediction = rbf_values @ params
    
    return prediction
# EVOLVE-BLOCK-END
