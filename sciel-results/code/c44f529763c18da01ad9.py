"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 145.0 # 5s safety margin

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check time
    if time.time() - _START_TIME > _TIMEOUT:
        return np.zeros_like(x)

    p = params
    
    # Chebyshev polynomials for x
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2 * x * Tx1 - Tx0
    Tx3 = 2 * x * Tx2 - Tx1
    
    # Chebyshev polynomials for y
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2 * y * Ty1 - Ty0
    Ty3 = 2 * y * Ty2 - Ty1
    
    # Basis functions ordered by total degree
    # Degree 0:
    # p[0] * T0(x) * T0(y)
    
    # Degree 1:
    # p[1] * T1(x) * T0(y)
    # p[2] * T0(x) * T1(y)
    
    # Degree 2:
    # p[3] * T2(x) * T0(y)
    # p[4] * T1(x) * T1(y)
    # p[5] * T0(x) * T2(y)
    
    # Degree 3:
    # p[6] * T3(x) * T0(y)
    # p[7] * T2(x) * T1(y)
    # p[8] * T1(x) * T2(y)
    # p[9] * T0(x) * T3(y)
    
    result = (p[0] * Tx0 * Ty0 + 
              p[1] * Tx1 * Ty0 + 
              p[2] * Tx0 * Ty1 + 
              p[3] * Tx2 * Ty0 + 
              p[4] * Tx1 * Ty1 + 
              p[5] * Tx0 * Ty2 + 
              p[6] * Tx3 * Ty0 + 
              p[7] * Tx2 * Ty1 + 
              p[8] * Tx1 * Ty2 + 
              p[9] * Tx0 * Ty3)
    
    return result
# EVOLVE-BLOCK-END
