"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    
    This model uses a hybrid approach:
    1. Physics-inspired basis functions (Duffing, Van der Pol, etc.) scaled to unit range.
    2. A constant offset term.
    
    The inputs are normalized to improve conditioning for the BFGS optimizer:
      xn = x / 0.2
      vn = v / 0.35
    
    Basis functions:
      0: 1                          (Offset)
      1: xn                         (Linear restoring)
      2: xn^3                       (Cubic restoring)
      3: vn                         (Linear damping)
      4: xn^2 * vn                  (Van der Pol damping)
      5: xn * vn                    (Gyroscopic coupling)
      6: vn^3                       (Cubic velocity)
      7: sin(t)                     (Drive in-phase)
      8: cos(t)                     (Drive quadrature)
      9: xn * sin(t)                (Parametric excitation)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalization
    xn = x / 0.2
    vn = v / 0.35

    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn

    # Trig terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    # Basis functions
    b0 = 1.0
    b1 = xn
    b2 = xn3
    b3 = vn
    b4 = xn2 * vn
    b5 = xn * vn
    b6 = vn3
    b7 = sin_t
    b8 = cos_t
    b9 = xn * sin_t

    # Linear combination
    return (p0 * b0 + 
            p1 * b1 + 
            p2 * b2 + 
            p3 * b3 + 
            p4 * b4 + 
            p5 * b5 + 
            p6 * b6 + 
            p7 * b7 + 
            p8 * b8 + 
            p9 * b9)
# EVOLVE-BLOCK-END
