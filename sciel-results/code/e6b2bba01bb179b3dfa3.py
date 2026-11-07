"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Radial Basis Function (RBF) Network.
    
    Structure:
    - 2 RBF centers in 2D space: (cx1, cy1) and (cx2, cy2).
    - 2 Widths (sigmas): sigma1, sigma2.
    - 2 Output weights: w1, w2.
    - 1 Global bias: b.
    
    Total parameters: 9 (using 9 of the 10 available).
    
    Model:
    f(x, y) = b + w1 * exp(-((x - cx1)^2 + (y - cy1)^2) / (2 * sigma1^2)) 
             + w2 * exp(-((x - cx2)^2 + (y - cy2)^2) / (2 * sigma2^2))
    
    This approach allows the optimizer (BFGS) to move the centers to regions
    of high variation in the target function and adjust widths to capture local
    features. This is structurally different from a fixed polynomial basis,
    as it can represent localized non-linearities more efficiently if the
    target function has localized structure.
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Extract parameters
    # Order: cx1, cy1, sigma1, w1, cx2, cy2, sigma2, w2, b
    cx1, cy1, sigma1, w1, cx2, cy2, sigma2, w2, b = params[:9]
    
    # Ensure widths are positive to avoid division by zero or negative variance
    # Using a small epsilon to prevent singularity
    eps = 1e-6
    s1 = max(abs(sigma1), eps)
    s2 = max(abs(sigma2), eps)
    
    # Compute squared distances
    dx1 = x - cx1
    dy1 = y - cy1
    dist1_sq = dx1 * dx1 + dy1 * dy1
    
    dx2 = x - cx2
    dy2 = y - cy2
    dist2_sq = dx2 * dx2 + dy2 * dy2
    
    # Compute RBF activations
    # exp(-dist^2 / (2 * sigma^2))
    rbf1 = np.exp(-dist1_sq / (2.0 * s1 * s1))
    rbf2 = np.exp(-dist2_sq / (2.0 * s2 * s2))
    
    # Combine
    result = b + w1 * rbf1 + w2 * rbf2
    
    return result
# EVOLVE-BLOCK-END
