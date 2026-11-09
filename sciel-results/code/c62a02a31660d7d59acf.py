"""Symbolic regression with a mixed polynomial and trigonometric basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixed basis: polynomial terms + trigonometric terms.
    
    Basis functions (10 total, matching 10 params):
    0: 1
    1: x
    2: y
    3: x^2
    4: xy
    5: y^2
    6: sin(pi * x)
    7: sin(pi * y)
    8: sin(pi * x) * sin(pi * y)
    9: cos(pi * x) * cos(pi * y)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Trigonometric terms
    sin_pi_x = np.sin(np.pi * x)
    sin_pi_y = np.sin(np.pi * y)
    cos_pi_x = np.cos(np.pi * x)
    cos_pi_y = np.cos(np.pi * y)
    
    # Construct the linear combination
    pred = (p0 + 
            p1 * x + 
            p2 * y + 
            p3 * x2 + 
            p4 * xy + 
            p5 * y2 + 
            p6 * sin_pi_x + 
            p7 * sin_pi_y + 
            p8 * sin_pi_x * sin_pi_y + 
            p9 * cos_pi_x * cos_pi_y)
            
    return pred
# EVOLVE-BLOCK-END
