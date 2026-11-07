"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter linear-in-params basis for symbolic regression on [-1,1]x[-1,1].
    Basis: 1, x, y, x^2, y^2, xy, sin(x), sin(y), cos(x), cos(y)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    return (p0 + p1 * x + p2 * y + p3 * x**2 + p4 * y**2 +
            p5 * x * y + p6 * np.sin(x) + p7 * np.sin(y) +
            p8 * np.cos(x) + p9 * np.cos(y))
# EVOLVE-BLOCK-END
