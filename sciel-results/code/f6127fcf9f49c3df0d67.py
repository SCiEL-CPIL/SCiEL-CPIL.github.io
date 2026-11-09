"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a trigonometric and interaction basis.
    Basis: 1, x, v, sin(x), sin(v), cos(x), cos(v), x*sin(v), v*sin(x), sin(x+v)
    """
    # Compute basis terms
    sin_x = np.sin(x)
    sin_v = np.sin(v)
    cos_x = np.cos(x)
    cos_v = np.cos(v)
    sin_xv = np.sin(x + v)
    
    # Construct the linear combination
    # p0: constant
    # p1: x
    # p2: v
    # p3: sin(x)
    # p4: sin(v)
    # p5: cos(x)
    # p6: cos(v)
    # p7: x * sin(v)
    # p8: v * sin(x)
    # p9: sin(x + v)
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = v
    term3 = sin_x
    term4 = sin_v
    term5 = cos_x
    term6 = cos_v
    term7 = x * sin_v
    term8 = v * sin_x
    term9 = sin_xv

    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
