"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a Van der Pol-inspired nonlinear basis to capture limit-cycle dynamics.
    Basis: 1, x, v, x^3, v^2, x*v, x^5, v^3, x^2*v, x^4
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    v2 = v * v
    v3 = v2 * v
    
    return (p0 + 
            p1 * x + 
            p2 * v + 
            p3 * x3 + 
            p4 * v2 + 
            p5 * x * v + 
            p6 * x5 + 
            p7 * v3 + 
            p8 * x2 * v + 
            p9 * x4)
# EVOLVE-BLOCK-END
