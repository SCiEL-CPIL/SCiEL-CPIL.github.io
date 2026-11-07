"""Symbolic regression: 10-parameter Chebyshev basis for 2D function recovery."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev(x, n):
    """Compute Chebyshev polynomial T_n(x) for n = 0,1,2,3,4."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 1:
        return x.copy()
    elif n == 2:
        return 2.0 * x * x - 1.0
    elif n == 3:
        return 4.0 * x * x * x - 3.0 * x
    elif n == 4:
        return 8.0 * x**4 - 8.0 * x**2 + 1.0
    else:
        return np.ones_like(x, dtype=float)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Chebyshev polynomial basis on [-1,1]^2.
    
    Features:
      0: T0(x)*T0(y) = 1           (constant)
      1: T1(x)                    (linear x)
      2: T1(y)                    (linear y)
      3: T2(x)                    (quadratic x)
      4: T2(y)                    (quadratic y)
      5: T3(x)                    (cubic x)
      6: T3(y)                    (cubic y)
      7: T4(x)                    (quartic x)
      8: T4(y)                    (quartic y)
      9: T1(x)*T1(y)              (bilinear interaction)
    """
    p = np.asarray(params, dtype=float)
    
    T0x = np.ones_like(x, dtype=float)
    T1x = x
    T2x = 2.0 * x * x - 1.0
    T3x = 4.0 * x**3 - 3.0 * x
    T4x = 8.0 * x**4 - 8.0 * x**2 + 1.0
    
    T0y = np.ones_like(y, dtype=float)
    T1y = y
    T2y = 2.0 * y * y - 1.0
    T3y = 4.0 * y**3 - 3.0 * y
    T4y = 8.0 * y**4 - 8.0 * y**2 + 1.0
    
    pred = (
        p[0] * T0x * T0y +
        p[1] * T1x +
        p[2] * T1y +
        p[3] * T2x +
        p[4] * T2y +
        p[5] * T1x * T1y +
        p[6] * T2x * T1y +
        p[7] * T1x * T2y +
        p[8] * T3x +
        p[9] * T3y
    )
    
    return pred
# EVOLVE-BLOCK-END
