"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with Gaussian kernels on a 3x3 grid covering [-1,1]^2.
    This includes the four corners, the four edge midpoints, and the origin.
    params: length-10 array.
      params[0..8]: weights for the 9 RBF centers.
      params[9]: global bias offset.
    """
    # Centers: 3x3 grid in [-1, 1]^2
    # x coordinates: -1, 0, 1
    # y coordinates: -1, 0, 1
    cx = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    cy = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    
    # Fixed bandwidth for Gaussian kernels
    # Chosen to allow localization while maintaining smoothness across the domain
    sigma = 0.7
    
    # Compute squared distances from each input point to each center
    # x, y are arrays of shape (n,)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    dx = x[:, np.newaxis] - cx[np.newaxis, :]  # (n, 9)
    dy = y[:, np.newaxis] - cy[np.newaxis, :]  # (n, 9)
    dist_sq = dx ** 2 + dy ** 2  # (n, 9)
    
    # Gaussian RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / (2.0 * sigma ** 2))  # (n, 9)
    
    # Weights for the 9 centers
    weights = params[0:9]
    
    # Global bias
    bias = params[9]
    
    # Prediction: sum of weighted RBFs + bias
    pred = rbf_values @ weights + bias
    
    return pred
# EVOLVE-BLOCK-END
