"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    
    Structural change: Internal normalization of inputs to improve conditioning
    for BFGS. The raw inputs t, x, v have very different scales (t in [30,50],
    x in [-0.2, 0.17], v in [-0.34, 0.36]). By normalizing them to roughly
    unit scale before applying the basis functions, the gradient landscape for
    the 10 parameters becomes better conditioned, allowing BFGS to converge
    to a better minimum within the fixed time budget.
    
    Normalization:
      tn = (t - 40) / 10   -> roughly [-1, 1]
      xn = x / 0.2         -> roughly [-1, 1]
      vn = v / 0.35        -> roughly [-1, 1]
    
    Model (in normalized variables):
      a = p0 * xn
        + p1 * xn^3            (Duffing cubic restoring)
        + p2 * vn              (linear damping)
        + p3 * vn^3            (cubic velocity saturation)
        + p4 * xn^2 * vn       (Van der Pol nonlinear damping)
        + p5 * xn * vn         (parametric / gyroscopic coupling)
        + p6 * tn              (slow linear drift)
        + p7 * sin(t)          (sinusoidal drive - use raw t for physical freq)
        + p8 * xn^2            (quadratic position correction)
        + p9                   (offset)
    
    Note: sin(t) uses raw t to preserve the physical oscillation frequency,
    but the amplitude is scaled by the normalized parameters.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to improve conditioning
    # t in [30, 50] -> center at 40, scale by 10 -> [-1, 1]
    tn = (t - 40.0) / 10.0
    # x in [-0.2056, 0.1749] -> scale by 0.2 -> roughly [-1, 1]
    xn = x / 0.2
    # v in [-0.3391, 0.3624] -> scale by 0.35 -> roughly [-1, 1]
    vn = v / 0.35

    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn

    # Duffing-type restoring (linear + cubic)
    restoring = p0 * xn + p1 * xn3

    # Van der Pol style damping: linear + cubic saturation + x^2*v
    damping = p2 * vn + p3 * vn3 + p4 * xn2 * vn

    # Parametric / gyroscopic coupling
    coupling = p5 * xn * vn

    # Time-dependent terms (drift + sinusoidal drive)
    # Use raw t for sin to maintain physical frequency, but tn for drift
    time_terms = p6 * tn + p7 * np.sin(t)

    # Quadratic position correction and offset
    correction = p8 * xn2 + p9

    return restoring + damping + coupling + time_terms + correction
# EVOLVE-BLOCK-END
