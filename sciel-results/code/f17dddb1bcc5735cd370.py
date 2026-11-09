"""Rewritten program for symbolic regression with 3 inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _legendre_shifted(n, t):
    """
    Compute the n-th Legendre polynomial evaluated at t, shifted to domain [0.5, 2].
    We map t in [0.5, 2] to u in [-1, 1] via u = (t - 1.25) / 0.75.
    Then P_n(u) is the standard Legendre polynomial.
    We use the recurrence relation for numerical stability.
    """
    # Map t from [0.5, 2] to u in [-1, 1]
    # center = (0.5 + 2) / 2 = 1.25, half_width = (2 - 0.5) / 2 = 0.75
    u = (t - 1.25) / 0.75
    
    # Clamp u to [-1, 1] for safety, though inputs are guaranteed in [0.5, 2]
    u = np.clip(u, -1.0, 1.0)
    
    if n == 0:
        return np.ones_like(u)
    elif n == 1:
        return u
    
    P0 = np.ones_like(u)
    P1 = u.copy()
    for k in range(1, n):
        P2 = ((2 * k + 1) * u * P1 - k * P0) / (k + 1)
        P0 = P1
        P1 = P2
    return P1


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich basis of shifted Legendre polynomials
    and their products to approximate the hidden function.
    
    params has length 10. We allocate them as follows:
    We want a flexible model. Let's use a combination of:
    - Constant
    - Linear in x, y, z (using shifted Legendre P1)
    - Quadratic terms (using P2)
    - Interaction terms
    
    However, we only have 10 parameters.
    
    Let's define a basis of 10 functions:
    f0 = 1
    f1 = P1(x)
    f2 = P1(y)
    f3 = P1(z)
    f4 = P2(x)
    f5 = P2(y)
    f6 = P2(z)
    f7 = P1(x) * P1(y)
    f8 = P1(x) * P1(z)
    f9 = P1(y) * P1(z)
    
    This basis is orthonormal-ish (Legendre polys are orthogonal) and captures
    linear, quadratic, and bilinear interactions.
    """
    # Compute shifted Legendre polynomials up to order 2
    P1_x = _legendre_shifted(1, x)
    P1_y = _legendre_shifted(1, y)
    P1_z = _legendre_shifted(1, z)
    
    P2_x = _legendre_shifted(2, x)
    P2_y = _legendre_shifted(2, y)
    P2_z = _legendre_shifted(2, z)
    
    # Build the prediction
    # params[0] * f0 + params[1] * f1 + ... + params[9] * f9
    
    pred = (params[0] * 1.0 +
            params[1] * P1_x +
            params[2] * P1_y +
            params[3] * P1_z +
            params[4] * P2_x +
            params[5] * P2_y +
            params[6] * P2_z +
            params[7] * (P1_x * P1_y) +
            params[8] * (P1_x * P1_z) +
            params[9] * (P1_y * P1_z))
    
    return pred
# EVOLVE-BLOCK-END
