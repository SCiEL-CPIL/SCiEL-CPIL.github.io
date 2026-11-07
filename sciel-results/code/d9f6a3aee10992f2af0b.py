"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Damped nonlinear oscillator with:
    - Linear damping (v)
    - Van der Pol nonlinear damping (x^2 * v)
    - Duffing stiffness (x^3)
    - Linear stiffness (x)
    - Periodic driving (sin(t), cos(t))
    - Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear damping
    linear_damp = p0 * v
    
    # Van der Pol nonlinear damping
    vdp_damp = p1 * x * x * v
    
    # Duffing nonlinear stiffness
    duffing = p2 * x * x * x
    
    # Linear stiffness
    linear_stiff = p3 * x
    
    # Periodic driving
    drive_sin = p4 * np.sin(t)
    drive_cos = p5 * np.cos(t)
    
    # Quadratic velocity term (for asymmetry)
    quad_v = p6 * v * v
    
    # Cross term x*v
    cross = p7 * x * v
    
    # t^2 term (slow drift)
    t_quad = p8 * t * t
    
    # Constant
    offset = p9
    
    return linear_damp + vdp_damp + duffing + linear_stiff + drive_sin + drive_cos + quad_v + cross + t_quad + offset
# EVOLVE-BLOCK-END
