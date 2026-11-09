"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with Gaussian kernels centered at the four corners of [-1,1]^2 and the origin.
    params: length-10 array.
      params[0..4]: weights for the 5 RBF centers.
      params[5..8]: (reserved for future use, currently unused but kept for interface stability)
      params[9]: global bias offset.
    """
    # Centers: four corners and the origin
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    # Fixed bandwidth for Gaussian kernels
    sigma = 1.0
    
    # Compute squared distances from each input point to each center
    # x, y are arrays of shape (n,)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    dx = x[:, np.newaxis] - cx[np.newaxis, :]  # (n, 5)
    dy = y[:, np.newaxis] - cy[np.newaxis, :]  # (n, 5)
    dist_sq = dx ** 2 + dy ** 2  # (n, 5)
    
    # Gaussian RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / (2.0 * sigma ** 2))  # (n, 5)
    
    # Weights for the 5 centers
    weights = params[0:5]
    
    # Global bias
    bias = params[9]
    
    # Prediction: sum of weighted RBFs + bias
    pred = rbf_values @ weights + bias
    
    return pred
# EVOLVE-BLOCK-END
