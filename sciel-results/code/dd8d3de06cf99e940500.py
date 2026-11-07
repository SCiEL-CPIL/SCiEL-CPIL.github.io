# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y in [-1, 1]^2.
    
    Strategy: Radial Basis Function (RBF) expansion.
    We use 9 Gaussian RBFs centered on a 3x3 grid covering the domain,
    plus 1 constant term (bias). This provides a flexible local approximation
    capability that is superior to low-degree global bases for general smooth
    functions.
    
    The 10 parameters correspond to:
    1-9: Weights for the 9 Gaussian RBFs
    10:  Weight for the constant term (bias)
    
    Centers are at {-0.8, 0, 0.8} x {-0.8, 0, 0.8}.
    Sigma (width) is fixed at 0.6 to balance local flexibility and smoothness.
    """
    p = params
    
    # Define the 9 centers for the RBFs
    # Using a 3x3 grid to cover [-1, 1]x[-1, 1]
    x_coords = [-0.8, 0.0, 0.8]
    y_coords = [-0.8, 0.0, 0.8]
    
    # Pre-compute squared distances for efficiency
    # x and y are 1D arrays of the same length
    # We need to compute ||(x,y) - (cx,cy)||^2 for each center
    
    # Unpack parameters
    # p[0] to p[8] are RBF weights
    # p[9] is the bias
    
    # Initialize output with the bias term
    out = np.full_like(x, p[9], dtype=np.float64)
    
    sigma_sq = 2 * (0.6 ** 2) # 2 * sigma^2 for the exponent
    
    idx = 0
    for cy in y_coords:
        for cx in x_coords:
            # Compute squared distance: (x - cx)^2 + (y - cy)^2
            dx = x - cx
            dy = y - cy
            dist_sq = dx * dx + dy * dy
            
            # Gaussian RBF: exp(-dist_sq / (2 * sigma^2))
            rbf_val = np.exp(-dist_sq / sigma_sq)
            
            out += p[idx] * rbf_val
            idx += 1
            
    return out
# EVOLVE-BLOCK-END
