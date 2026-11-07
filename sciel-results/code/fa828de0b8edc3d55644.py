"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: quadratic polynomial in normalized coords plus
    three Fourier-like nonlinear terms. This adds capacity for
    oscillatory or saturating behavior while keeping the linear-in-params
    structure so BFGS stays fast and well-conditioned.
    
    Basis (10 terms):
    0: 1
    1: u_x
    2: u_y
    3: u_z
    4: u_x^2 + u_y^2 + u_z^2  (isotropic quadratic)
    5: u_x * u_y + u_x * u_z + u_y * u_z  (isotropic cross)
    6: sin(3.0 * u_x) * cos(2.0 * u_y)
    7: sin(2.0 * u_y) * cos(3.0 * u_z)
    8: sin(3.0 * u_z) * cos(2.0 * u_x)
    9: u_x * u_y * u_z  (trilinear interaction)
    """
    p = params
    
    # Normalize inputs to [-1, 1]
    u_x = (x - 1.25) / 0.75
    u_y = (y - 1.25) / 0.75
    u_z = (z - 1.25) / 0.75
    
    quad_iso = u_x * u_x + u_y * u_y + u_z * u_z
    cross_iso = u_x * u_y + u_x * u_z + u_y * u_z
    trig1 = np.sin(3.0 * u_x) * np.cos(2.0 * u_y)
    trig2 = np.sin(2.0 * u_y) * np.cos(3.0 * u_z)
    trig3 = np.sin(3.0 * u_z) * np.cos(2.0 * u_x)
    tri = u_x * u_y * u_z
    
    return (p[0] +
            p[1] * u_x +
            p[2] * u_y +
            p[3] * u_z +
            p[4] * quad_iso +
            p[5] * cross_iso +
            p[6] * trig1 +
            p[7] * trig2 +
            p[8] * trig3 +
            p[9] * tri)
# EVOLVE-BLOCK-END
