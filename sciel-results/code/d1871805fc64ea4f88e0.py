"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stable symbolic regression using scaled polynomial basis.
    
    The previous complex model likely suffered from ill-conditioning due to
    high powers of x in [-2, 2]. By scaling x to u = x/2 in [-1, 1], we
    ensure that all basis functions (powers of u) are bounded in [-1, 1],
    leading to a well-conditioned design matrix. This allows BFGS to
    converge reliably to the optimal parameters, effectively reducing to
    the linear model if the target is linear, but retaining capacity for
    higher-order smooth functions.
    """
    p = params
    
    # Scale input to [-1, 1] to improve numerical stability of polynomial basis
    u = x * 0.5
    
    # Construct polynomial basis up to degree 9
    # Using Horner's method or simple powers. Simple powers are fine for n=10.
    b = np.empty((10, len(x)), dtype=np.float64)
    
    b[0] = 1.0
    b[1] = u
    b[2] = u * u
    b[3] = b[2] * u
    b[4] = b[2] * b[2]
    b[5] = b[4] * u
    b[6] = b[3] * b[3]
    b[7] = b[6] * u
    b[8] = b[4] * b[4]
    b[9] = b[8] * u
    
    # Compute linear combination
    # Using matrix multiplication for efficiency and clarity
    # b is (10, N), p is (10,)
    return b @ p
# EVOLVE-BLOCK-END
