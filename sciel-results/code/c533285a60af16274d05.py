"""Nonlinear oscillator model with physics-informed bounded basis functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a damped nonlinear oscillator.
    
    Uses a bounded nonlinear basis that captures:
    - Linear restoring force (x)
    - Cubic softening/hardening (x^3)
    - Saturation via tanh
    - Linear damping (v)
    - Quadratic drag (v*|v|)
    - Coupling terms (x*v, x^2, v^2)
    - Constant offset
    
    10 parameters total.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Position terms
    x2 = x * x
    x3 = x2 * x
    
    # Velocity terms
    v2 = v * v
    
    # Bounded nonlinear terms
    tanh_x = np.tanh(x)
    sin_x = np.sin(x)
    
    # Quadratic drag: v * |v|
    v_abs = np.abs(v)
    quad_drag = v * v_abs
    
    # Coupling terms
    xv = x * v
    
    # Construct the equation with 10 parameters
    # p0: linear restoring force
    # p1: cubic position (softening/hardening)
    # p2: tanh saturation of position
    # p3: sine of position (periodic restoring)
    # p4: linear damping
    # p5: quadratic drag
    # p6: coupling x*v
    # p7: quadratic position
    # p8: quadratic velocity
    # p9: constant offset
    result = (
        p0 * x
        + p1 * x3
        + p2 * tanh_x
        + p3 * sin_x
        + p4 * v
        + p5 * quad_drag
        + p6 * xv
        + p7 * x2
        + p8 * v2
        + p9
    )
    
    return result
# EVOLVE-BLOCK-END
