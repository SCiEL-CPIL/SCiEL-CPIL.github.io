"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a basis of linear, exponential decay, and logarithmic terms.
    This structural change moves away from polynomial bases (which failed) to a mixed transcendental basis.
    """
    p = params
    
    # Basis functions
    b0 = 1.0
    b1 = x
    b2 = y
    b3 = z
    b4 = np.exp(-x)
    b5 = np.exp(-y)
    b6 = np.exp(-z)
    b7 = np.log(x)
    b8 = np.log(y)
    b9 = np.log(z)
    
    return (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
# EVOLVE-BLOCK-END
