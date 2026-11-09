"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs from [0.5, 2.0] to [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    
    # Legendre polynomials on [-1, 1]
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3t^2 - 1)
    
    # For u
    Pu1 = u
    Pu2 = 0.5 * (3.0 * u * u - 1.0)
    
    # For v
    Pv1 = v
    Pv2 = 0.5 * (3.0 * v * v - 1.0)
    
    # For w
    Pw1 = w
    Pw2 = 0.5 * (3.0 * w * w - 1.0)
    
    # 10 basis functions:
    # 0: constant (P0)
    # 1: P1(u)
    # 2: P1(v)
    # 3: P1(w)
    # 4: P2(u)
    # 5: P2(v)
    # 6: P2(w)
    # 7: P1(u) * P1(v)
    # 8: P1(u) * P1(w)
    # 9: P1(v) * P1(w)
    
    result = (p[0]
              + p[1] * Pu1
              + p[2] * Pv1
              + p[3] * Pw1
              + p[4] * Pu2
              + p[5] * Pv2
              + p[6] * Pw2
              + p[7] * Pu1 * Pv1
              + p[8] * Pu1 * Pw1
              + p[9] * Pv1 * Pw1)
    
    return result
# EVOLVE-BLOCK-END
