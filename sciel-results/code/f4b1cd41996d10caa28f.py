"""Symbolic regression with 2 inputs using a matched 10-param degree-3 bivariate polynomial."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a degree-3 bivariate polynomial with 10 coefficients.
    
    The basis functions are:
    0: 1
    1: x
    2: y
    3: x^2
    4: x*y
    5: y^2
    6: x^3
    7: x^2*y
    8: x*y^2
    9: y^3
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    # Extract parameters
    p = params[:10]
    
    # Compute basis functions
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    
    # Build prediction: linear combination of basis functions
    pred = (p[0] + 
            p[1] * x + 
            p[2] * y + 
            p[3] * x2 + 
            p[4] * xy + 
            p[5] * y2 + 
            p[6] * x3 + 
            p[7] * x2 * y + 
            p[8] * x * y2 + 
            p[9] * y3)
    
    return pred
# EVOLVE-BLOCK-END
