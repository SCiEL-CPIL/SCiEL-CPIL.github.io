import numpy as np
from scipy.optimize import minimize

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a full cubic polynomial basis.
    This basis covers all monomials up to degree 3 in the scaled inputs X and V.
    This is a robust general-purpose approximation for smooth nonlinear functions.
    
    Inputs are scaled to improve BFGS conditioning.
    """
    # Scale inputs to roughly [-1, 1]
    X = x / 0.6
    V = v / 0.3
    
    # Compute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    # Cross terms
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Build basis matrix (10 terms)
    # The 10 terms of the full cubic polynomial:
    # p0: 1
    # p1: X
    # p2: V
    # p3: X^2
    # p4: X*V
    # p5: V^2
    # p6: X^3
    # p7: X^2*V
    # p8: X*V^2
    # p9: V^3
    
    t0 = np.ones_like(x)
    t1 = X
    t2 = V
    t3 = X2
    t4 = XV
    t5 = V2
    t6 = X3
    t7 = X2V
    t8 = XV2
    t9 = V3
    
    T = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])
    
    return T @ params
