"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 basis functions.
    
    Basis functions:
      0: 1 (constant)
      1: x
      2: x^2
      3: x^3
      4: sin(x)
      5: cos(x)
      6: sin(2x)
      7: cos(2x)
      8: exp(-0.5 * x^2)
      9: tanh(x)
    
    This is linear in parameters, so BFGS converges to the global minimum
    (least-squares solution) for the given basis.
    """
    p = params
    b0 = 1.0
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    b8 = np.exp(-0.5 * b2)
    b9 = np.tanh(x)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
