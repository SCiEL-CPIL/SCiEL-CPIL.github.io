"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    n = len(x)
    if n == 0:
        return np.array([])

    # Use a rich basis of 10 functions that can capture a wide variety of shapes
    # on [-2, 2]:
    # - Constant, linear, quadratic, cubic, quartic, quintic (polynomial up to degree 5)
    # - sin(x), cos(x), sin(2x), cos(2x) (Fourier terms)
    # This gives 10 basis functions total.
    
    # Polynomial basis
    p0 = np.ones(n)
    p1 = x.copy()
    p2 = x ** 2
    p3 = x ** 3
    p4 = x ** 4
    p5 = x ** 5

    # Fourier basis
    f1 = np.sin(x)
    f2 = np.cos(x)
    f3 = np.sin(2.0 * x)
    f4 = np.cos(2.0 * x)

    # Combine all 10 basis functions with the 10 parameters
    result = (params[0] * p0 +
              params[1] * p1 +
              params[2] * p2 +
              params[3] * p3 +
              params[4] * p4 +
              params[5] * p5 +
              params[6] * f1 +
              params[7] * f2 +
              params[8] * f3 +
              params[9] * f4)

    return result
# EVOLVE-BLOCK-END
