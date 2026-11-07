"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Trigonometric harmonic basis with fixed non-integer frequency multipliers
    # 10 basis functions corresponding to 10 params
    
    p = params
    
    # Basis 0: constant (intercept)
    b0 = np.ones_like(x)
    
    # Basis 1: sin(2*x)
    b1 = np.sin(2.0 * x)
    
    # Basis 2: cos(3*y)
    b2 = np.cos(3.0 * y)
    
    # Basis 3: sin(4*z)
    b3 = np.sin(4.0 * z)
    
    # Basis 4: sin(x + y)
    b4 = np.sin(x + y)
    
    # Basis 5: cos(y + z)
    b5 = np.cos(y + z)
    
    # Basis 6: sin(z + x)
    b6 = np.sin(z + x)
    
    # Basis 7: cos(1.5 * x)
    b7 = np.cos(1.5 * x)
    
    # Basis 8: sin(2.5 * y)
    b8 = np.sin(2.5 * y)
    
    # Basis 9: cos(3.5 * z)
    b9 = np.cos(3.5 * z)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
