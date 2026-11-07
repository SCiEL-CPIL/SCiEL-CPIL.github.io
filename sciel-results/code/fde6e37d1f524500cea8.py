"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Full quadratic polynomial basis in x, y, z
    # Terms: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    # params[0] corresponds to the constant term, etc.
    # Order: [1, x, y, z, x^2, y^2, z^2, xy, xz, yz]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    
    return p0 + p1 * x + p2 * y + p3 * z + p4 * x2 + p5 * y2 + p6 * z2 + p7 * xy + p8 * xz + p9 * yz
# EVOLVE-BLOCK-END
