import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Autonomous Van der Pol / Duffing oscillator model.
    No explicit time-driven terms; intrinsic limit-cycle dynamics.
    
    Basis functions (10 parameters):
    0: constant offset
    1: linear stiffness (x)
    2: cubic stiffness (x^3) - Duffing hardening/softening
    3: linear damping (v)
    4: cubic velocity damping (v^3) - saturation
    5: quadratic velocity damping (v^2)
    6: Van der Pol damping (x^2 * v)
    7: coupling (x * v)
    8: quadratic position (x^2)
    9: quintic position (x^5) - higher-order stiffness
    """
    p = params
    
    # Compute basis terms
    x2 = x ** 2
    x3 = x ** 3
    x5 = x ** 5
    v2 = v ** 2
    v3 = v ** 3
    
    term_const = np.ones_like(x)
    term_x = x
    term_v = v
    term_vdp = x2 * v
    term_xv = x * v
    
    return (p[0] * term_const +
            p[1] * term_x +
            p[2] * x3 +
            p[3] * term_v +
            p[4] * v3 +
            p[5] * v2 +
            p[6] * term_vdp +
            p[7] * term_xv +
            p[8] * x2 +
            p[9] * x5)
