"""Symbolic regression: 10-parameter symmetric Chebyshev basis for 2D function recovery."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s limit


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter symmetric Chebyshev polynomial basis on [-1,1]^2.
    
    Features (symmetric up to degree 3):
      0: 1                       (constant)
      1: T1(x)                   (linear x)
      2: T1(y)                   (linear y)
      3: T2(x)                   (quadratic x)
      4: T2(y)                   (quadratic y)
      5: T1(x)*T1(y)             (bilinear interaction)
      6: T3(x)                   (cubic x)
      7: T3(y)                   (cubic y)
      8: T2(x)*T1(y)             (quadratic-x, linear-y interaction)
      9: T1(x)*T2(y)             (linear-x, quadratic-y interaction)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple prediction to avoid crashing, though this shouldn't happen
        # as the function is very fast.
        return np.zeros_like(x, dtype=float)

    p = np.asarray(params, dtype=float)
    
    # Chebyshev polynomials for x
    T1x = x
    T2x = 2.0 * x * x - 1.0
    T3x = 4.0 * x**3 - 3.0 * x
    
    # Chebyshev polynomials for y
    T1y = y
    T2y = 2.0 * y * y - 1.0
    T3y = 4.0 * y**3 - 3.0 * y
    
    pred = (
        p[0] +
        p[1] * T1x +
        p[2] * T1y +
        p[3] * T2x +
        p[4] * T2y +
        p[5] * T1x * T1y +
        p[6] * T3x +
        p[7] * T3y +
        p[8] * T2x * T1y +
        p[9] * T1x * T2y
    )
    
    return pred
# EVOLVE-BLOCK-END
