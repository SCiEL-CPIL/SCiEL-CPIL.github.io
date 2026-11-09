"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Basis functions for symbolic regression with 10 parameters.
    
    Features:
    p0: 1
    p1: x
    p2: y
    p3: z
    p4: x*y
    p5: x*z
    p6: y*z
    p7: x*y*z
    p8: x**2
    p9: y**2
    """
    f0 = np.ones_like(x, dtype=float)
    f1 = x
    f2 = y
    f3 = z
    f4 = x * y
    f5 = x * z
    f6 = y * z
    f7 = x * y * z
    f8 = x * x
    f9 = y * y
    
    pred = (params[0] * f0 +
            params[1] * f1 +
            params[2] * f2 +
            params[3] * f3 +
            params[4] * f4 +
            params[5] * f5 +
            params[6] * f6 +
            params[7] * f7 +
            params[8] * f8 +
            params[9] * f9)
    
    return pred
# EVOLVE-BLOCK-END
