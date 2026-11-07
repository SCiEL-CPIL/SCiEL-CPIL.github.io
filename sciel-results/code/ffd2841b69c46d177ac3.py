# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier + polynomial basis model with 10 parameters.
    
    Basis functions (10 total, linear in params):
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: sin(3x)
    9: cos(3x)
    
    This gives a rich global basis that can fit:
    - Polynomials up to degree 3
    - Periodic functions with fundamental and harmonics
    - Smooth functions via Fourier approximation
    """
    p = params
    
    # Polynomial terms
    t0 = np.ones_like(x)
    t1 = x
    t2 = x * x
    t3 = x * x * x
    
    # Fourier terms
    t4 = np.sin(x)
    t5 = np.cos(x)
    t6 = np.sin(2.0 * x)
    t7 = np.cos(2.0 * x)
    t8 = np.sin(3.0 * x)
    t9 = np.cos(3.0 * x)
    
    y = (p[0] * t0 +
         p[1] * t1 +
         p[2] * t2 +
         p[3] * t3 +
         p[4] * t4 +
         p[5] * t5 +
         p[6] * t6 +
         p[7] * t7 +
         p[8] * t8 +
         p[9] * t9)
    
    return y
# EVOLVE-BLOCK-END
