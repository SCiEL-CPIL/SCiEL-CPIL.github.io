"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable product basis with multiplicative interactions.
    f(x,y,z) = (a0 + a1*x + a2*x^2) * (b0 + b1*y + b2*y^2) * (c0 + c1*z + c2*z^2) + d
    
    10 params: [a0, a1, a2, b0, b1, b2, c0, c1, c2, d]
    """
    p = params
    a0, a1, a2 = p[0], p[1], p[2]
    b0, b1, b2 = p[3], p[4], p[5]
    c0, c1, c2 = p[6], p[7], p[8]
    d = p[9]
    
    # Compute each factor
    fx = a0 + a1 * x + a2 * x**2
    fy = b0 + b1 * y + b2 * y**2
    fz = c0 + c1 * z + c2 * z**2
    
    # Product plus constant offset
    return fx * fy * fz + d
# EVOLVE-BLOCK-END
