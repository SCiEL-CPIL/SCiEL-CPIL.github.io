# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physically-motivated basis.
    
    Basis terms (10 params):
    p0: 1 (constant/driving offset)
    p1: x (linear restoring)
    p2: x^3 (Duffing nonlinear restoring)
    p3: v (linear damping)
    p4: v^3 (cubic damping)
    p5: x*v (velocity-dependent stiffness)
    p6: x^2 (quadratic nonlinearity)
    p7: v^2 (quadratic velocity term)
    p8: x^2*v (cross nonlinearity)
    p9: x^3*v (higher-order cross term)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    term_0 = p0
    term_1 = p1 * x
    term_2 = p2 * x3
    term_3 = p3 * v
    term_4 = p4 * v3
    term_5 = p5 * x * v
    term_6 = p6 * x2
    term_7 = p7 * v2
    term_8 = p8 * x2 * v
    term_9 = p9 * x3 * v
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
# EVOLVE-BLOCK-END
