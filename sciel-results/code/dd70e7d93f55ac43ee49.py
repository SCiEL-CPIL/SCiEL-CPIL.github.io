"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 fixed basis functions.
    This makes the problem linear in params, so BFGS can find the global optimum.
    Basis functions are chosen to span a wide variety of smooth functions on [0.1, 6].
    """
    # Precompute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = x ** 5
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.exp(-x)
    b9 = 1.0 / x
    
    # Linear combination
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
# EVOLVE-BLOCK-END
