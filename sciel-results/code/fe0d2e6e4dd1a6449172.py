"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator with scale-normalized basis terms.
    Each basis term is normalized to have ~unit magnitude in the training range,
    making the optimization landscape well-conditioned for BFGS from all-ones init.
    Training ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624], t in [30, 50]
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    x2v = x2 * v
    xv = x * v
    
    # Normalize each basis term to ~unit scale based on typical magnitudes
    # x ~ 0.2, x^3 ~ 0.008, v ~ 0.35, x^2*v ~ 0.015, x*v ~ 0.07
    # x^2 ~ 0.04, v^2 ~ 0.12, sin/cos ~ 1
    
    result = (
        p0 * 0.1
        + p1 * (x / 0.2)
        + p2 * (x3 / 0.008)
        + p3 * (v / 0.35)
        + p4 * (x2v / 0.015)
        + p5 * np.sin(t)
        + p6 * np.cos(t)
        + p7 * (x2 / 0.04)
        + p8 * (v2 / 0.12)
        + p9 * (xv / 0.07)
    )
    
    return result
# EVOLVE-BLOCK-END
