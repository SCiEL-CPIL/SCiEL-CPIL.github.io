"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    
    pred = (
        p[0] +
        p[1] * x +
        p[2] * y +
        p[3] * x2 +
        p[4] * xy +
        p[5] * y2 +
        p[6] * x3 +
        p[7] * x2y +
        p[8] * xy2 +
        p[9] * y3
    )
    return pred
# EVOLVE-BLOCK-END
