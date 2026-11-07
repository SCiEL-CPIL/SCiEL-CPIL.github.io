"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: combines polynomial and trigonometric terms to capture
    both smooth and oscillatory behavior in the unknown function.
    
    The 10 parameters are fitted by BFGS. The basis includes:
    - Constant, linear, and quadratic polynomial terms
    - Trigonometric terms (sin, cos) to capture periodic behavior
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Trigonometric terms with different frequencies
    # Using frequency 1 and 2 to capture oscillatory behavior
    sin_x = np.sin(np.pi * x)
    cos_x = np.cos(np.pi * x)
    sin_y = np.sin(np.pi * y)
    cos_y = np.cos(np.pi * y)
    
    # Basis functions: mix of polynomial and trig
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x2
    b4 = y2
    b5 = x * y
    b6 = sin_x
    b7 = sin_y
    b8 = cos_x * cos_y
    b9 = x3 + y3  # combined cubic term
    
    # Linear combination
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
