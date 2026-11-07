"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Basis: 1, x, x^3, v, x^2*v, sin(t), cos(t), x*sin(t), x^2, v^3
    Covers Duffing (x, x^3), Van der Pol damping (v, x^2*v),
    sinusoidal driving (sin(t), cos(t)), parametric forcing (x*sin(t)),
    and cubic velocity damping (v^3).
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    x2v = x2 * v
    
    st = np.sin(t)
    ct = np.cos(t)
    xst = x * st
    
    result = (
        p0
        + p1 * x
        + p2 * x3
        + p3 * v
        + p4 * x2v
        + p5 * st
        + p6 * ct
        + p7 * xst
        + p8 * x2
        + p9 * v3
    )
    
    return result
# EVOLVE-BLOCK-END
