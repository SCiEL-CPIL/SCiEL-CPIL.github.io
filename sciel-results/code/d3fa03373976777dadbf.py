"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Complete bivariate polynomial of degree <= 3 (10 terms)
    # Basis: 1, x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    pred = (params[0]
            + params[1] * x
            + params[2] * y
            + params[3] * x2
            + params[4] * x * y
            + params[5] * y2
            + params[6] * x3
            + params[7] * x2 * y
            + params[8] * x * y2
            + params[9] * y3)
    
    return pred
# EVOLVE-BLOCK-END
