"""Symbolic regression: nonlinear oscillator equation."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Structural Change: Uses a mixed basis of low-order polynomials and tanh non-linearities.
    This basis is designed to capture both local curvature (polynomials) and global
    saturation/boundedness (tanh) typical of physical oscillators, potentially helping
    the BFGS optimizer find a better global minimum than a pure polynomial basis.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute basis terms
    x2 = x * x
    v2 = v * v
    
    # Non-linear terms
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    # Basis:
    # p0: constant
    # p1: linear x
    # p2: linear v
    # p3: quadratic x
    # p4: quadratic v
    # p5: tanh(x) (nonlinear restoring)
    # p6: tanh(v) (nonlinear damping)
    # p7: x * tanh(v)
    # p8: v * tanh(x)
    # p9: x^3 (higher order restoring)
    
    return (p0 + p1 * x + p2 * v + p3 * x2 + p4 * v2 + 
            p5 * tanh_x + p6 * tanh_v + p7 * x * tanh_v + 
            p8 * v * tanh_x + p9 * x2 * x)
# EVOLVE-BLOCK-END
