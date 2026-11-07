"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-of-sums basis with 10 parameters.
    
    f(x,y,z) = (a0 + a1*x + a2*y + a3*z) * (b0 + b1*x + b2*y + b3*z) + c0
    
    This expands to:
    - Constant: a0*b0 + c0
    - Linear: a0*b1 + a1*b0, a0*b2 + a2*b0, a0*b3 + a3*b0
    - Bilinear: a1*b1, a1*b2, a1*b3, a2*b1, a2*b2, a2*b3, a3*b1, a3*b2, a3*b3
    
    Parameters:
    0-3: coefficients for first linear factor (a0, a1, a2, a3)
    4-7: coefficients for second linear factor (b0, b1, b2, b3)
    8:   additional constant offset (c0)
    9:   scale factor for the product term
    
    This captures additive linear trends, interactions, and bilinear dependencies
    through the product of two affine hyperplanes.
    """
    p = params
    # First linear factor
    L1 = p[0] + p[1] * x + p[2] * y + p[3] * z
    # Second linear factor
    L2 = p[4] + p[5] * x + p[6] * y + p[7] * z
    # Product of sums with scale and offset
    return p[8] * L1 * L2 + p[9]
# EVOLVE-BLOCK-END
