"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a cubic polynomial basis in position (x) and velocity (v).
    Basis: 1, x, v, x^2, xv, v^2, x^3, x^2v, xv^2, v^3
    """
    # Compute basis terms
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    # Construct the linear combination
    # p0: constant
    # p1: x
    # p2: v
    # p3: x^2
    # p4: x*v
    # p5: v^2
    # p6: x^3
    # p7: x^2*v
    # p8: x*v^2
    # p9: v^3
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = v
    term3 = x2
    term4 = x * v
    term5 = v2
    term6 = x3
    term7 = x2 * v
    term8 = x * v2
    term9 = v3

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
