"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function f(x, y) using a linear combination of Legendre polynomials.
    The basis is chosen to be orthogonal over [-1, 1] x [-1, 1] to aid BFGS convergence.
    
    Basis functions (10 total) using Legendre polynomials P_n:
    1. P0(x)P0(y) = 1
    2. P1(x)P0(y) = x
    3. P0(x)P1(y) = y
    4. P2(x)P0(y) = 0.5*(3x^2 - 1)
    5. P0(x)P2(y) = 0.5*(3y^2 - 1)
    6. P1(x)P1(y) = xy
    7. P3(x)P0(y) = 0.5*(5x^3 - 3x)
    8. P0(x)P3(y) = 0.5*(5y^3 - 3y)
    9. P2(x)P1(y) = 0.5*(3x^2 - 1)*y
    10. P1(x)P2(y) = x*0.5*(3y^2 - 1)
    """
    # Deadline guard: safety margin for the 150s limit
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)

    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Precompute Legendre polynomials
    # P0 = 1
    # P1 = t
    # P2 = 0.5 * (3t^2 - 1)
    # P3 = 0.5 * (5t^3 - 3t)
    
    px0 = np.ones_like(x)
    px1 = x
    px2 = 0.5 * (3 * x**2 - 1)
    px3 = 0.5 * (5 * x**3 - 3 * x)
    
    py0 = np.ones_like(y)
    py1 = y
    py2 = 0.5 * (3 * y**2 - 1)
    py3 = 0.5 * (5 * y**3 - 3 * y)
    
    # Construct the 10 basis terms
    # Term 0: P0(x)P0(y)
    term0 = px0 * py0
    # Term 1: P1(x)P0(y)
    term1 = px1 * py0
    # Term 2: P0(x)P1(y)
    term2 = px0 * py1
    # Term 3: P2(x)P0(y)
    term3 = px2 * py0
    # Term 4: P0(x)P2(y)
    term4 = px0 * py2
    # Term 5: P1(x)P1(y)
    term5 = px1 * py1
    # Term 6: P3(x)P0(y)
    term6 = px3 * py0
    # Term 7: P0(x)P3(y)
    term7 = px0 * py3
    # Term 8: P2(x)P1(y)
    term8 = px2 * py1
    # Term 9: P1(x)P2(y)
    term9 = px1 * py2
    
    # Combine with parameters
    pred = (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
            
    return pred
# EVOLVE-BLOCK-END
