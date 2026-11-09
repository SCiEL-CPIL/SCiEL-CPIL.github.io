"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1]
    # Inputs are in [0.5, 2.0]
    # center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(u)
    
    # 2-4. Linear terms (P1)
    b1 = u
    b2 = v
    b3 = w
    
    # 5-7. Quadratic terms (P2)
    # P2(t) = 0.5 * (3t^2 - 1)
    b4 = 0.5 * (3.0 * u * u - 1.0)
    b5 = 0.5 * (3.0 * v * v - 1.0)
    b6 = 0.5 * (3.0 * w * w - 1.0)
    
    # 8-10. Interaction terms
    b7 = u * v
    b8 = u * w
    b9 = v * w
    
    # Combine
    return (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
# EVOLVE-BLOCK-END
