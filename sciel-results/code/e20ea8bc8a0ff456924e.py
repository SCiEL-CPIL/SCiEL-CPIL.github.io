"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a 3rd-order polynomial in x and v to capture nonlinear dynamics.
    params: [bias, x, v, x^2, v^2, x*v, x^3, v^3, x^2*v, x*v^2]
    """
    p = params
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    return (p[0] + 
            p[1] * x + 
            p[2] * v + 
            p[3] * x2 + 
            p[4] * v2 + 
            p[5] * x * v + 
            p[6] * x3 + 
            p[7] * v3 + 
            p[8] * x2 * v + 
            p[9] * x * v2)
# EVOLVE-BLOCK-END
