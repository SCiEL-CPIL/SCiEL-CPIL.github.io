"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven nonlinear oscillator.
    
    Model:
    a = p0*sin(t) + p1*cos(t)       (Periodic drive)
      + p2*x + p3*x^3               (Duffing stiffness: linear + cubic)
      + p4*v + p5*x*v + p6*v^2      (Damping: linear, cross, quadratic)
      + p7*x^2*v                     (Van der Pol damping)
      + p8*x^2                       (Quadratic position term)
      + p9                           (Constant offset)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers and trig functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Construct the equation
    a = p0 * sin_t + p1 * cos_t
    a = a + p2 * x + p3 * x3
    a = a + p4 * v + p5 * x * v + p6 * v2
    a = a + p7 * x2 * v
    a = a + p8 * x2
    a = a + p9
    
    return a
# EVOLVE-BLOCK-END
