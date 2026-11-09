"""Symbolic regression for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    Uses a rich basis of nonlinear terms in x, v, and t to capture
    nonlinear dynamics, including:
    - Linear and nonlinear restoring forces (x, x^2, x^3, x^4)
    - Linear and nonlinear damping (v, v^2, v^3, x*v, x^2*v)
    - Time-dependent driving (t, sin(t), cos(t), t*sin(t), t*cos(t))
    - Cross terms
    
    params: array of 10 fitted constants
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Nonlinear position-dependent terms (restoring force)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Nonlinear velocity-dependent terms (damping)
    v2 = v * v
    v3 = v2 * v
    
    # Cross terms
    xv = x * v
    x2v = x2 * v
    
    # Time-dependent terms (driving force)
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    t_sin_t = t * sin_t
    t_cos_t = t * cos_t
    
    # Combine: 
    # p0: linear spring
    # p1: cubic spring
    # p2: quartic spring
    # p3: quadratic spring
    # p4: linear damping
    # p5: quadratic damping (v^2)
    # p6: cubic damping
    # p7: cross term x*v
    # p8: sinusoidal drive
    # p9: time-linear drive
    
    result = (p0 * x + p1 * x3 + p2 * x4 + p3 * x2 +
              p4 * v + p5 * v2 + p6 * v3 + p7 * xv +
              p8 * sin_t + p9 * t)
    
    return result
# EVOLVE-BLOCK-END
