"""Rewritten program for symbolic regression using a multiplicative parametric form."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a multiplicative parametric form.
    
    The model is a product of two affine linear terms:
        f(x, y, z) = (a1*x + b1*y + c1*z + d1) * (a2*x + b2*y + c2*z + d2)
    
    This expands to 10 parameters:
        a1*a2*x^2 + (a1*b2 + b1*a2)*x*y + (a1*c2 + c1*a2)*x*z + 
        (b1*c2 + c1*b2)*y*z + b1*b2*y^2 + c1*c2*z^2 + 
        (a1*d2 + d1*a2)*x + (b1*d2 + d1*b2)*y + (c1*d2 + d1*c2)*z + d1*d2
    
    This form can exactly represent quadratic polynomials with a specific 
    rank-2 structure, and can approximate rational functions when one factor 
    is small. It's a structural departure from additive basis combinations.
    
    The 10 params are: [a1, b1, c1, d1, a2, b2, c2, d2, scale, offset]
    where the actual model is: scale * (a1*x + b1*y + c1*z + d1) * (a2*x + b2*y + c2*z + d2) + offset
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Extract parameters
    a1, b1, c1, d1, a2, b2, c2, d2, scale, offset = params
    
    # Compute the two affine factors
    factor1 = a1 * x + b1 * y + c1 * z + d1
    factor2 = a2 * x + b2 * y + c2 * z + d2
    
    # Compute the prediction
    pred = scale * factor1 * factor2 + offset
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
