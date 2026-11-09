"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression basis with 10 parameters using Legendre polynomials.
    
    Legendre polynomials on [-1,1] are orthogonal and well-conditioned,
    which helps BFGS converge robustly. We use a complete degree-3 basis
    in 2D (10 terms) to span all smooth functions up to cubic order.
    
    Basis functions:
    P0(x)P0(y), P1(x)P0(y), P0(x)P1(y), P2(x)P0(y), P0(x)P2(y),
    P1(x)P1(y), P3(x)P0(y), P0(x)P3(y), P2(x)P1(y), P1(x)P2(y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Legendre polynomials evaluated at x and y
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = (3t^2 - 1)/2
    # P3(t) = (5t^3 - 3t)/2
    
    P0_x = np.ones_like(x)
    P1_x = x
    P2_x = (3.0 * x * x - 1.0) / 2.0
    P3_x = (5.0 * x * x * x - 3.0 * x) / 2.0
    
    P0_y = np.ones_like(y)
    P1_y = y
    P2_y = (3.0 * y * y - 1.0) / 2.0
    P3_y = (5.0 * y * y * y - 3.0 * y) / 2.0
    
    # 10 basis terms: all combinations of {P0,P1,P2,P3}(x) * {P0,P1,P2,P3}(y)
    # with total degree <= 3
    b1 = P0_x * P0_y
    b2 = P1_x * P0_y
    b3 = P0_x * P1_y
    b4 = P2_x * P0_y
    b5 = P0_x * P2_y
    b6 = P1_x * P1_y
    b7 = P3_x * P0_y
    b8 = P0_x * P3_y
    b9 = P2_x * P1_y
    b10 = P1_x * P2_y
    
    return (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[8] * b9 + 
            params[9] * b10)
# EVOLVE-BLOCK-END
