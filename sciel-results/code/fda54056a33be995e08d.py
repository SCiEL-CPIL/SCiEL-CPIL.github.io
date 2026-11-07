"""
Radial Basis Function (RBF) network with a learnable center and width, 
plus a global polynomial basis.
This structure targets localized non-linear features which global bases 
approximate poorly, while the polynomial terms handle global trends.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 120.0  # 120 seconds limit, keep safety margin

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF Network + Polynomial Basis.
    
    Parameters (10 total):
    p0: center_x
    p1: center_y
    p2: width (sigma)
    p3: rbf_weight
    p4: coeff_x
    p5: coeff_y
    p6: coeff_xy
    p7: coeff_x2
    p8: coeff_y2
    p9: bias
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Check deadline
    if time.time() > _DEADLINE - 5:
        return np.zeros_like(x)

    cx = params[0]
    cy = params[1]
    sigma = params[2]
    w_rbf = params[3]
    
    wx = params[4]
    wy = params[5]
    wxy = params[6]
    wx2 = params[7]
    wy2 = params[8]
    b = params[9]
    
    # Compute RBF term: w_rbf * exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
    # Clamp sigma to avoid division by zero or extreme values
    sigma_clamped = np.clip(sigma, 0.01, 10.0)
    
    dx = x - cx
    dy = y - cy
    dist_sq = dx**2 + dy**2
    
    # Avoid overflow in exp
    # If dist_sq is large and sigma is small, exp(-large) -> 0
    # If dist_sq is small and sigma is small, exp(-small) -> 1
    # The term inside exp is -dist_sq / (2 * sigma^2)
    # Max value is 0 (when dist=0). Min value is -inf.
    # So exp is always <= 1. No overflow risk from exp itself, but 
    # division by zero if sigma=0. Clamped sigma handles this.
    
    rbf_val = np.exp(-dist_sq / (2.0 * sigma_clamped**2))
    
    # Compute polynomial terms
    poly_val = wx * x + wy * y + wxy * x * y + wx2 * x**2 + wy2 * y**2 + b
    
    return w_rbf * rbf_val + poly_val
# EVOLVE-BLOCK-END
