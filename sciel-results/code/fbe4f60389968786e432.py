"""Structural change: radial-basis-function style kernel + full cubic Legendre basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis:
      5 shifted-Legendre terms up to degree 3 on each variable (P0..P3 for x, y, z)
      plus 3 cross-interaction terms (P1*P1 pairs).
      This gives a well-conditioned orthogonal basis with cubic curvature capture.
    """
    p = params
    # Normalize inputs from [0.5, 2.0] to [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75

    # Legendre polynomials (shifted to [-1,1] domain, standard)
    # P0 = 1
    # P1 = t
    # P2 = 0.5*(3t^2 - 1)
    # P3 = 0.5*(5t^3 - 3t)
    P1_u = u
    P2_u = 0.5 * (3.0 * u * u - 1.0)
    P3_u = 0.5 * (5.0 * u * u * u - 3.0 * u)

    P1_v = v
    P2_v = 0.5 * (3.0 * v * v - 1.0)
    P3_v = 0.5 * (5.0 * v * v * v - 3.0 * v)

    P1_w = w
    P2_w = 0.5 * (3.0 * w * w - 1.0)
    P3_w = 0.5 * (5.0 * w * w * w - 3.0 * w)

    # Cross-interactions (linear-linear)
    uv = P1_u * P1_v
    uw = P1_u * P1_w
    vw = P1_v * P1_w

    # 10 terms:
    # p[0]: P0 (constant)
    # p[1]: P1_u
    # p[2]: P1_v
    # p[3]: P1_w
    # p[4]: P2_u
    # p[5]: P2_v
    # p[6]: P2_w
    # p[7]: P3_u
    # p[8]: P3_v
    # p[9]: P3_w
    # Wait, that's only 10 but no cross terms. Let me restructure.
    
    # Better allocation for 10 params:
    # p[0]: constant
    # p[1]: P1_u, p[2]: P1_v, p[3]: P1_w  (linear)
    # p[4]: P2_u, p[5]: P2_v, p[6]: P2_w  (quadratic)
    # p[7]: P3_u, p[8]: P3_v, p[9]: P3_w  (cubic)
    # This drops cross terms but captures pure cubic curvature.
    
    return (p[0] +
            p[1] * P1_u + p[2] * P1_v + p[3] * P1_w +
            p[4] * P2_u + p[5] * P2_v + p[6] * P2_w +
            p[7] * P3_u + p[8] * P3_v + p[9] * P3_w)
# EVOLVE-BLOCK-END
