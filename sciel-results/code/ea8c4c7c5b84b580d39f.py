"""Rewritten program: RBF basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form bivariate function using Radial Basis Functions (RBFs).
    
    We use 9 RBF centers on a 3x3 grid in [-1, 1]^2 and 1 constant term.
    This provides a flexible, non-polynomial approximation space that is
    often better conditioned for optimization than high-degree polynomials.
    
    Basis functions:
    b0 = 1 (constant offset)
    b1..b9 = exp(-||[x, y] - c_i||^2 / (2 * sigma^2))
    
    Centers c_i are at:
    (-1, -1), (-1, 0), (-1, 1)
    ( 0, -1), ( 0, 0), ( 0, 1)
    ( 1, -1), ( 1, 0), ( 1, 1)
    
    Sigma (width) is fixed at 0.5.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Define centers
    centers = [
        (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
        ( 0.0, -1.0), ( 0.0, 0.0), ( 0.0, 1.0),
        ( 1.0, -1.0), ( 1.0, 0.0), ( 1.0, 1.0)
    ]
    
    sigma2 = 0.5
    denom = 2.0 * sigma2 * sigma2
    
    # Compute RBF values
    # b0 is the constant term
    b0 = np.ones_like(x)
    
    # b1 to b9 correspond to the 9 centers
    # We accumulate the weighted sum directly to avoid storing large intermediate arrays if possible,
    # but for clarity and safety with BFGS, we compute the basis matrix or sum terms.
    # Given N=200, computing 9 exponentials is cheap.
    
    pred = params[0] * b0
    
    for i, (cx, cy) in enumerate(centers):
        # Distance squared
        dx = x - cx
        dy = y - cy
        dist_sq = dx * dx + dy * dy
        # RBF value
        rbf_val = np.exp(-dist_sq / denom)
        pred += params[i + 1] * rbf_val
        
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.mean(y) * np.ones_like(x)
        
    return pred
# EVOLVE-BLOCK-END
