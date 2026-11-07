"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    # Scale input to [-1, 1] for better numerical stability of high-order polynomials
    z = x / 2.0
    
    # Polynomial of degree 9
    # y = p0 + p1*z + p2*z^2 + ... + p9*z^9
    y = np.zeros_like(z)
    power = np.ones_like(z)
    for i in range(10):
        y += params[i] * power
        power *= z
        
    return y
# EVOLVE-BLOCK-END
