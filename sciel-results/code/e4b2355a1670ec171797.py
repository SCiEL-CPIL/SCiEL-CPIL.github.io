"""Rewritten program for symbolic regression with multiplicative parametric form."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a multiplicative parametric form.
    
    The model is a product of two affine linear terms:
    (a1*x + b1*y + c1*z + d1) * (a2*x + b2*y + c2*z + d2)
    
    This can exactly represent:
    - Products like x*y, x*z, y*z
    - Sums like x+y+z (if one factor is constant)
    - More complex bilinear structures
    
    params (10 values) are interpreted as:
    [a1, b1, c1, d1, a2, b2, c2, d2, scale, offset]
    
    prediction = scale * (a1*x + b1*y + c1*z + d1) * (a2*x + b2*y + c2*z + d2) + offset
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Extract parameters
    a1, b1, c1, d1, a2, b2, c2, d2, scale, offset = params
    
    # Compute the two affine linear terms
    term1 = a1 * x + b1 * y + c1 * z + d1
    term2 = a2 * x + b2 * y + c2 * z + d2
    
    # Multiply and apply scale and offset
    pred = scale * term1 * term2 + offset
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
