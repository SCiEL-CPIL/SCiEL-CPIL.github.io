"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-constrained rational function for nonlinear oscillator.
    Acceleration a = N(x, v, t) / D(x, v, t)
    
    The numerator captures the restoring force and damping:
      N = p0 + p1*x + p2*x^3 + p3*v + p4*x^2*v + p5*sin(t) + p6*cos(t)
    
    The denominator models softening/hardening:
      D = 1 + p7*x^2 + p8*v^2 + p9*x*v
    
    This structure naturally captures saturation and nonlinear coupling.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    x2v = x2 * v
    xv = x * v
    
    # Numerator: polynomial + periodic terms
    numerator = (
        p0
        + p1 * x
        + p2 * x3
        + p3 * v
        + p4 * x2v
        + p5 * np.sin(t)
        + p6 * np.cos(t)
    )
    
    # Denominator: 1 + quadratic terms (ensures D != 0 in typical range)
    # The quadratic terms are small in the training range:
    # x^2 ~ 0.04, v^2 ~ 0.12, x*v ~ 0.07
    denominator = 1.0 + p7 * x2 + p8 * v2 + p9 * xv
    
    # Guard against division by zero
    denominator = np.where(np.abs(denominator) < 1e-10, np.sign(denominator) * 1e-10 + 1e-10, denominator)
    
    result = numerator / denominator
    
    return result
# EVOLVE-BLOCK-END
