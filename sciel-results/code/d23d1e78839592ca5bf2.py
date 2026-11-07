"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a polynomial basis in normalized inputs to capture nonlinear dynamics.
    Normalization helps the optimizer converge by keeping feature scales similar.
    """
    # Normalize inputs based on known ranges to improve conditioning
    # t in [30, 50] -> center 40, scale 10
    t_norm = (t - 40.0) / 10.0
    # x in [-0.2, 0.17] -> scale 0.2
    x_norm = x / 0.2
    # v in [-0.34, 0.36] -> scale 0.35
    v_norm = v / 0.35

    p = params

    # Basis functions
    # 1. Constant
    b0 = 1.0
    # 2. Time linear
    b1 = t_norm
    # 3. Position linear
    b2 = x_norm
    # 4. Velocity linear
    b3 = v_norm
    # 5. Position quadratic (stiffness nonlinearity)
    b4 = x_norm ** 2
    # 6. Velocity quadratic (drag nonlinearity)
    b5 = v_norm ** 2
    # 7. Cross term x*v
    b6 = x_norm * v_norm
    # 8. Position cubic (Duffing)
    b7 = x_norm ** 3
    # 9. Time-modulated stiffness
    b8 = t_norm * x_norm
    # 10. Time-modulated damping
    b9 = t_norm * v_norm

    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
