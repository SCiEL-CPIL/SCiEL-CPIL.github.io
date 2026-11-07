"""Symbolic regression: recover hidden function from 2 inputs using Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Chebyshev polynomial basis for 2D function recovery on [-1,1]^2.
    
    The basis spans total degree up to 3 using Chebyshev polynomials:
      Degree 0: 1
      Degree 1: T1(x), T1(y)
      Degree 2: T2(x), T2(y), T1(x)*T1(y)
      Degree 3: T3(x), T3(y), T2(x)*T1(y), T1(x)*T2(y)
    
    This provides a well-conditioned, orthogonal-ish basis for smooth bivariate
    functions on the unit square, which is ideal for BFGS optimization.
    
    params: array of length 10, weights for the 10 basis functions.
    Returns: array of predictions, one per point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Chebyshev polynomials for x
    T1x = x
    T2x = 2.0 * x * x - 1.0
    T3x = 4.0 * x * x * x - 3.0 * x
    
    # Chebyshev polynomials for y
    T1y = y
    T2y = 2.0 * y * y - 1.0
    T3y = 4.0 * y * y * y - 3.0 * y
    
    # Build 10 basis functions
    b0 = np.ones_like(x)          # constant
    b1 = T1x                      # linear x
    b2 = T1y                      # linear y
    b3 = T2x                      # quadratic x
    b4 = T2y                      # quadratic y
    b5 = T1x * T1y                # bilinear interaction xy
    b6 = T3x                      # cubic x
    b7 = T3y                      # cubic y
    b8 = T2x * T1y                # x^2 * y interaction
    b9 = T1x * T2y                # x * y^2 interaction
    
    # Linear combination
    pred = (p[0] * b0 +
            p[1] * b1 +
            p[2] * b2 +
            p[3] * b3 +
            p[4] * b4 +
            p[5] * b5 +
            p[6] * b6 +
            p[7] * b7 +
            p[8] * b8 +
            p[9] * b9)
    
    # Ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return pred
# EVOLVE-BLOCK-END
