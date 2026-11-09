"""Symbolic regression with 2 inputs - complete cubic polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a complete cubic polynomial basis in (x, y).
    
    With 10 parameters, we can exactly represent any polynomial of degree <= 3
    in two variables. The basis is:
    [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
    
    This gives a rich, smooth, and well-conditioned basis for fitting
    smooth target functions via BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure at least 10 params (pad with zeros if needed)
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Complete cubic polynomial: 
    # p0 + p1*x + p2*y + p3*x^2 + p4*x*y + p5*y^2 + p6*x^3 + p7*x^2*y + p8*x*y^2 + p9*y^3
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    result = (p0 
              + p1 * x 
              + p2 * y 
              + p3 * x2 
              + p4 * x * y 
              + p5 * y2 
              + p6 * x3 
              + p7 * x2 * y 
              + p8 * x * y2 
              + p9 * y3)
    
    return result
# EVOLVE-BLOCK-END
