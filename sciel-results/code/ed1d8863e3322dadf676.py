# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    
    Uses a flexible combination of polynomial terms in x and v, plus time-dependent
    forcing, to capture nonlinear oscillator dynamics.
    
    Features:
    p0: constant
    p1: linear x
    p2: linear v
    p3: x^2
    p4: v^2
    p5: x*v
    p6: x^3
    p7: v^3
    p8: t*x (time-varying coupling)
    p9: t*v (time-varying damping correction)
    """
    t = np.asarray(t)
    x = np.asarray(x)
    v = np.asarray(v)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Construct basis functions
    f0 = np.ones_like(x)
    f1 = x
    f2 = v
    f3 = x ** 2
    f4 = v ** 2
    f5 = x * v
    f6 = x ** 3
    f7 = x ** 2 * v
    f8 = t * x
    f9 = t * v
    
    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 +
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
