"""Symbolic regression: bivariate polynomial basis with 10 params."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate polynomial of total degree <= 3 with 10 coefficients.
    Basis: 1, x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3
    params[0..9] are the coefficients for each basis function.
    """
    p = params
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    return (
        p[0]
        + p[1] * x
        + p[2] * y
        + p[3] * x2
        + p[4] * x * y
        + p[5] * y2
        + p[6] * x3
        + p[7] * x2 * y
        + p[8] * x * y2
        + p[9] * y3
    )
# EVOLVE-BLOCK-END
