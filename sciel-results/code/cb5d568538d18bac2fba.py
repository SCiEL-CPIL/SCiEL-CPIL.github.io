"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    We use a rich basis: polynomials up to degree 3, sin, cos, exp(-x^2/2), tanh, and a rational-like term.
    The basis is scaled to keep values in a reasonable range for BFGS starting at 1s.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params

    # Polynomial terms (scaled by 1/3 to keep magnitude ~1 on [-3,3])
    u = x / 3.0
    u2 = u * u
    u3 = u2 * u

    # Trigonometric
    sx = np.sin(x)
    cx = np.cos(x)

    # Gaussian bump
    gx = np.exp(-0.5 * x * x)

    # Hyperbolic tangent
    tx = np.tanh(x)

    # A decaying oscillatory term
    dx = np.exp(-0.3 * x * x) * np.sin(2.0 * x)

    # A rational-like smooth term (1/(1+x^2))
    rx = 1.0 / (1.0 + x * x)

    # Combine: 10 terms
    # p0: constant
    # p1: linear
    # p2: quadratic
    # p3: cubic
    # p4: sin
    # p5: cos
    # p6: gaussian
    # p7: tanh
    # p8: decaying sine
    # p9: rational
    return (p[0] +
            p[1] * u +
            p[2] * u2 +
            p[3] * u3 +
            p[4] * sx +
            p[5] * cx +
            p[6] * gx +
            p[7] * tx +
            p[8] * dx +
            p[9] * rx)
# EVOLVE-BLOCK-END
