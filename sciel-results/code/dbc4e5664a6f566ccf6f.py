import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a 3rd-order polynomial basis.
    Inputs are normalized to O(1) range to improve numerical conditioning
    for the BFGS optimizer.
    
    Basis: All monomials X^i * V^j where i + j <= 3.
    X = x / S_x, V = v / S_v
    """
    p = params
    
    # Normalization constants based on training range
    # x in [-0.5728, 0.5275] -> scale by max abs ~ 0.5728
    # v in [-0.1754, 0.2821] -> scale by max abs ~ 0.2821
    S_x = 0.5728
    S_v = 0.2821
    
    X = x / S_x
    V = v / S_v
    
    # Compute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    # Construct the expression using the 10 parameters
    # Degree 0
    term0 = p[0]
    
    # Degree 1
    term1 = p[1] * X + p[2] * V
    
    # Degree 2
    term2 = p[3] * X2 + p[4] * X * V + p[5] * V2
    
    # Degree 3
    term3 = p[6] * X3 + p[7] * X2 * V + p[8] * X * V2 + p[9] * V3
    
    return term0 + term1 + term2 + term3
