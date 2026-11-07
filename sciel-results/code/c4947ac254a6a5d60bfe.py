"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based approximation with 10 fixed, evenly spaced centers and a single learnable width.
    
    params[0:9] : weights for the 10 RBF kernels
    params[9]   : log-width parameter (log of the RBF width sigma)
    
    The RBFs are centered at evenly spaced points in [-3, 3].
    The width is exp(params[9]), clipped to a reasonable range to avoid numerical issues.
    """
    # Fixed centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Learnable width: use exp of param to ensure positivity, clip to avoid extremes
    log_sigma = params[9]
    # Clip log_sigma to keep sigma in a reasonable range, e.g., [0.1, 10.0]
    log_sigma_clipped = np.clip(log_sigma, np.log(0.1), np.log(10.0))
    sigma = np.exp(log_sigma_clipped)
    
    # Compute RBF values: exp(-((x - c_i)^2) / (2 * sigma^2))
    # x shape: (n,), centers shape: (10,)
    # We want a matrix of shape (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-diff ** 2 / (2.0 * sigma ** 2))
    
    # Linear combination with weights params[0:9]
    weights = params[0:10]
    pred = rbf_values @ weights
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
