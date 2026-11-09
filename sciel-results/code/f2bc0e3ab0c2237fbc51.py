import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 learnable parameters.
    Uses a rich basis of centered polynomials, trigonometric, and rational terms.
    """
    # Center and scale inputs to roughly [-1, 1]
    # Domain [0.5, 2.0], center 1.25, half-range 0.75
    xc = (x - 1.25) / 0.75
    yc = (y - 1.25) / 0.75
    zc = (z - 1.25) / 0.75
    
    # Basis functions - mix of polynomial, trig, and rational terms
    # b0: constant
    b0 = np.ones_like(x)
    
    # Linear terms
    b1 = xc
    b2 = yc
    b3 = zc
    
    # Quadratic terms
    b4 = xc * xc
    b5 = yc * yc
    b6 = zc * zc
    
    # Cross terms (selected for coverage)
    b7 = xc * yc
    b8 = xc * zc
    b9 = yc * zc
    
    # Combine with learnable parameters
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 +
            params[4] * b4 + params[5] * b5 + params[6] * b6 + params[7] * b7 +
            params[8] * b8 + params[9] * b9)
