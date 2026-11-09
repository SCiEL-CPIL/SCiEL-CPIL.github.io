"""
Strategy: Adaptive Gaussian Basis Functions with Learnable Centers and Scales.

This approach uses a mixture of 3 Gaussian basis functions.
Unlike Chebyshev polynomials, which are global and fixed, Gaussians can
localize to specific features of the function. By making the centers (mu)
and scales (sigma) learnable parameters, the BFGS optimizer can shift and
reshape the basis functions to match the target function's structure.

Params layout (length 10):
- params[0:3]: Weights (amplitudes) for the 3 Gaussians
- params[3:6]: Centers (mu) for the 3 Gaussians
- params[6:9]: Log-scales (log_sigma) for the 3 Gaussians. 
               Using log ensures sigma > 0 and improves optimization stability.
- params[9]:   Global bias (offset)

The function is:
y = bias + sum_{i=0}^{2} w_i * exp(-0.5 * ((x - mu_i) / sigma_i)^2)
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a mixture of 3 adaptive Gaussians.
    """
    # Extract parameters
    # Weights
    w0 = params[0]
    w1 = params[1]
    w2 = params[2]
    
    # Centers
    mu0 = params[3]
    mu1 = params[4]
    mu2 = params[5]
    
    # Log-scales (to ensure positive sigma)
    log_sigma0 = params[6]
    log_sigma1 = params[7]
    log_sigma2 = params[8]
    
    # Bias
    bias = params[9]
    
    # Compute sigmas with a floor to prevent division by zero or extreme values
    # exp(log_sigma) can be very small; floor at 1e-3 to keep gradients stable
    sigma0 = np.maximum(np.exp(log_sigma0), 1e-3)
    sigma1 = np.maximum(np.exp(log_sigma1), 1e-3)
    sigma2 = np.maximum(np.exp(log_sigma2), 1e-3)
    
    # Compute Gaussian terms
    # Using squared distance for numerical stability
    d0 = (x - mu0) / sigma0
    d1 = (x - mu1) / sigma1
    d2 = (x - mu2) / sigma2
    
    g0 = np.exp(-0.5 * d0 * d0)
    g1 = np.exp(-0.5 * d1 * d1)
    g2 = np.exp(-0.5 * d2 * d2)
    
    # Combine
    result = bias + w0 * g0 + w1 * g1 + w2 * g2
    
    return result
# EVOLVE-BLOCK-END
