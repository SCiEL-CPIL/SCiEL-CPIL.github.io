"""Symbolic regression: polynomial basis of degree 3 in x and y (10 terms)."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a complete polynomial basis of degree 3 in x and y.
    10 basis functions:
      1. 1
      2. x
      3. y
      4. x^2
      5. x*y
      6. y^2
      7. x^3
      8. x^2*y
      9. x*y^2
      10. y^3
    params: length-10 array of coefficients (fitted by BFGS).
    Returns an array with one prediction per point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute each basis term
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = x * y
    b5 = y * y
    b6 = x * x * x
    b7 = x * x * y
    b8 = x * y * y
    b9 = y * y * y

    # Linear combination
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)

    return result
# EVOLVE-BLOCK-END
