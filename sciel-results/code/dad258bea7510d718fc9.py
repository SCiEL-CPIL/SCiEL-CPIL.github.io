"""Bootstrap portfolio: Radial Basis Function (RBF) model with fixed centers."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Radial Basis Function (RBF) model.
    The function is approximated as a weighted sum of Gaussian bumps centered
    at fixed locations in the domain. This provides a localized basis that
    can capture features not well-represented by global polynomials.
    
    params: array of 10 weights for the 10 fixed RBF centers.
    """
    p = params
    
    # Fixed centers for the RBFs
    # 4 corners, 4 mid-edges, 1 center, 1 offset
    centers = np.array([
        [ 1.0,  1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [-1.0, -1.0],
        [ 0.0,  1.0],
        [ 0.0, -1.0],
        [ 1.0,  0.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 0.5,  0.5]
    ])
    
    # Width of the Gaussian
    sigma = 1.0
    sigma_sq_2 = 2.0 * sigma * sigma
    
    result = np.zeros_like(x)
    
    for i in range(10):
        cx, cy = centers[i]
        dx = x - cx
        dy = y - cy
        dist_sq = dx * dx + dy * dy
        kernel = np.exp(-dist_sq / sigma_sq_2)
        result += p[i] * kernel
        
    return result
# EVOLVE-BLOCK-END
