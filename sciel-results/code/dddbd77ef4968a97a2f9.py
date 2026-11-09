"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven nonlinear oscillator with quadratic and cubic terms.
    
    Structural change: replaced x^3 and v^3 with x^2 and v^2, and dropped
    sin(t)*x and cos(t)*x in favor of x*v^2 and v*x^2 interaction terms.
    
    Model structure:
    a = p0 + p1*sin(t) + p2*cos(t) + p3*x + p4*x^2 + p5*v + p6*v^2 + p7*x*v + p8*x^2*v + p9*v^2*x
    """
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    x2 = x * x
    v2 = v * v
    xv = x * v
    x2v = x2 * v
    v2x = v2 * x
    
    return (params[0] + 
            params[1] * sin_t + 
            params[2] * cos_t + 
            params[3] * x + 
            params[4] * x2 + 
            params[5] * v + 
            params[6] * v2 + 
            params[7] * xv + 
            params[8] * x2v + 
            params[9] * v2x)
# EVOLVE-BLOCK-END
