"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Van der Pol style nonlinear oscillator model with amplitude-dependent damping.
    
    Uses a nonlinear basis of terms to capture limit-cycle dynamics:
    {1, x, v, x^3, v^2, x*v, x^5, v^3, x^2*v, x^4}
    
    The model is:
    a = p0*1 + p1*x + p2*v + p3*x^3 + p4*v^2 + p5*x*v + p6*x^5 + p7*v^3 + p8*x^2*v + p9*x^4
    
    This captures:
    - Linear damping/driving (x, v)
    - Nonlinear stiffness (x^3, x^5, x^4)
    - Amplitude-dependent damping (v^2, v^3, x*v, x^2*v)
    - Constant offset (1)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    v2 = v * v
    v3 = v2 * v
    
    xv = x * v
    x2v = x2 * v
    
    return (p0 * np.ones_like(x) + 
            p1 * x + 
            p2 * v + 
            p3 * x3 + 
            p4 * v2 + 
            p5 * xv + 
            p6 * x5 + 
            p7 * v3 + 
            p8 * x2v + 
            p9 * x4)
# EVOLVE-BLOCK-END
