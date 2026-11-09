"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive RBF model with Gaussian kernels.
    Centers: Four corners of [-1, 1]^2 and the origin.
    params: length-10 array.
      params[0:5]: weights for the 5 centers.
      params[5:10]: log-bandwidth parameters for the 5 centers.
    """
    # Centers: 4 corners + origin
    cx = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    cy = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    
    # Extract parameters
    weights = params[0:5]
    log_sigmas = params[5:10]
    
    # Compute adaptive bandwidths
    # Map log_sigmas to positive sigmas.
    # Using exp ensures positivity. Scaling by 0.1 keeps it in a reasonable range for [-1,1]
    # when params start at 1 (sigma ~ 0.27).
    sigmas = 0.1 * np.exp(log_sigmas)
    
    # Compute squared distances
    # x, y are arrays of shape (n,)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    dx = x[:, np.newaxis] - cx[np.newaxis, :]  # (n, 5)
    dy = y[:, np.newaxis] - cy[np.newaxis, :]  # (n, 5)
    dist_sq = dx ** 2 + dy ** 2  # (n, 5)
    
    # Gaussian RBF values
    # rbf_values[i, j] = exp(-dist_sq[i, j] / (2 * sigma_j^2))
    # We need to divide by sigma_j^2 for each column j
    sigma_sq = sigmas ** 2
    rbf_values = np.exp(-dist_sq / (2.0 * sigma_sq[np.newaxis, :]))  # (n, 5)
    
    # Prediction: sum of weighted RBFs
    # The origin center with a large sigma acts as a bias term
    pred = rbf_values @ weights
    
    return pred
# EVOLVE-BLOCK-END
