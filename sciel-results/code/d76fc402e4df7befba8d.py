"""
Symbolic regression: Rational function with quadratic numerator and a 
smooth, strictly positive quadratic denominator.
This design avoids the non-smoothness and singularity issues of the 
standard rational form by squaring the denominator's variable part, 
ensuring the denominator is always >= 1. This provides a smooth 
landscape for BFGS optimization, which is critical for converging 
to a good local minimum with a non-convex model.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions for given x, y, params.
    
    params has length 10, corresponding to:
        p0, p1, p2, p3, p4, p5 (numerator coefficients)
        p6, p7, p8, p9 (denominator coefficients)
    
    The model is:
        (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y) /
        (1 + (p6*x + p7*y + p8*x^2 + p9*y^2)^2)
    
    This form is smooth everywhere and the denominator is bounded away 
    from zero, preventing numerical instabilities and providing a 
    well-behaved gradient for the optimizer.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute common terms
    x2 = x**2
    y2 = y**2
    xy = x * y
    
    # Compute numerator
    numerator = p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2 + p5 * xy
    
    # Compute the variable part of the denominator
    den_var = p6 * x + p7 * y + p8 * x2 + p9 * y2
    
    # Compute denominator: 1 + (den_var)^2 ensures D >= 1 and is smooth
    denominator = 1.0 + den_var**2
    
    # Compute predictions
    pred = numerator / denominator
    
    # Ensure the output is a finite array with correct shape
    # Since denominator >= 1, division is safe. 
    # If params are extremely large, pred might overflow, so we check finiteness.
    if not np.all(np.isfinite(pred)):
        # Fallback: if overflow occurred, clip inputs or params? 
        # Usually BFGS keeps params in a reasonable range if the loss is well-behaved.
        # If we get here, it's a numerical edge case.
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return pred
# EVOLVE-BLOCK-END
