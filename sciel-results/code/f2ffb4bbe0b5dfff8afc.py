"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator (Duffing-van der Pol structure).
    
    Structure: a = p0*x + p1*x^3 - p2*v - p3*v^3 + p4*x*v^2 + p5*v^2*x^2 
               + p6*x^2*v + p7*x^4 + p8*v^4 + p9
    
    The signs are chosen to reflect physical intuition:
    - p0: linear spring (negative for restoring)
    - p1: cubic nonlinearity (positive for hardening)
    - p2: linear damping (positive coefficient, negative sign in equation)
    - p3: cubic damping (positive coefficient, negative sign in equation)
    - p4-p8: cross terms
    - p9: constant bias
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    
    term0 = x
    term1 = x3
    term2 = v
    term3 = v3
    term4 = x * v2
    term5 = v2 * x2
    term6 = x2 * v
    term7 = x4
    term8 = v4
    term9 = np.ones_like(x)
    
    return (params[0] * term0 +
            params[1] * term1 -
            params[2] * term2 -
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
