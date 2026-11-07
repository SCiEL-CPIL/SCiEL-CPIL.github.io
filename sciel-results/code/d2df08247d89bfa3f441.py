"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a radial basis function (RBF) expansion.
    
    The model is:
    z = p[0] + sum_{i=1}^{9} p[i] * exp(-||[x,y] - c_i||^2 / (2*sigma^2))
    
    where c_i are 9 fixed centers on a 3x3 grid in [-1,1]^2, and sigma is fixed.
    This provides a flexible, smooth basis that can approximate many nonlinear functions.
    """
    p = params.astype(float)
    
    # Fixed RBF centers: 3x3 grid
    centers_x = np.array([-0.7, 0.0, 0.7])
    centers_y = np.array([-0.7, 0.0, 0.7])
    
    # Fixed bandwidth
    sigma = 0.5
    
    # Initialize output with constant term
    result = np.full_like(x, p[0], dtype=float)
    
    # Add each RBF term
    for i in range(9):
        cx = centers_x[i // 3]
        cy = centers_y[i % 3]
        
        # Compute squared distance from each point to center i
        dx = x - cx
        dy = y - cy
        dist_sq = dx * dx + dy * dy
        
        # RBF: exp(-dist_sq / (2 * sigma^2))
        rbf_val = np.exp(-dist_sq / (2.0 * sigma * sigma))
        
        result += p[i + 1] * rbf_val
    
    return result
# EVOLVE-BLOCK-END
