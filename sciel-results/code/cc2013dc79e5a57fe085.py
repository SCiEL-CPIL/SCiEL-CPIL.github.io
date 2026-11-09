"""Rewritten symbolic regression candidate: Rich mixed polynomial-trigonometric basis with higher-order terms."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s limit

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed set of 10 basis functions.
    params: length-10 array of coefficients (fitted by BFGS).
    Returns an array with one prediction per point.
    """
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        pass

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Richer basis functions:
    # 1. Constant
    # 2. x
    # 3. y
    # 4. x^2
    # 5. y^2
    # 6. x*y
    # 7. sin(x)*cos(y) - mixed trig
    # 8. cos(x)*sin(y) - mixed trig
    # 9. sin(x + y) - combined trig
    # 10. cos(x - y) - combined trig

    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = np.sin(x) * np.cos(y)
    b7 = np.cos(x) * np.sin(y)
    b8 = np.sin(x + y)
    b9 = np.cos(x - y)

    # Linear combination
    return (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
# EVOLVE-BLOCK-END
