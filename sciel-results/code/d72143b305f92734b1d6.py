"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    Uses a sparse cubic polynomial to capture Duffing and Van der Pol type nonlinearities
    while maintaining numerical stability for the optimizer.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Sparse cubic terms
    a = (p[0] * x +
         p[1] * v +
         p[2] * x**3 +
         p[3] * v**3 +
         p[4] * x * v +
         p[5] * x**2 * v +
         p[6] * x * v**2 +
         p[7] * x**3 * v +
         p[8] * x * v**3 +
         p[9])
    
    return a
# EVOLVE-BLOCK-END
