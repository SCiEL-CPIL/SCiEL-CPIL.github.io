"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis: Pure cubic polynomials in x, y, z
    # Terms: 1, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3
    # This assumes the function is separable or dominated by pure powers,
    # allowing higher-order curvature to be captured without the constraint of cross-terms.
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    y2 = y * y
    z2 = z * z
    
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    
    return p0 + p1 * x + p2 * y + p3 * z + p4 * x2 + p5 * y2 + p6 * z2 + p7 * x3 + p8 * y3 + p9 * z3
# EVOLVE-BLOCK-END
