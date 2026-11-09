"""Symbolic regression with 2 inputs: Cubic Polynomial Basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix.
    Strategy: Cubic polynomial basis.
    This covers all polynomial terms up to degree 3.
    This is a very strong basis for smooth functions on [-1, 1].
    """
    n = x.shape[0]
    
    # Precompute powers
    x2 = x * x
    y2 = y * y
    xy = x * y
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2

    f0 = np.ones(n)
    f1 = x
    f2 = y
    f3 = x2
    f4 = y2
    f5 = xy
    f6 = x3
    f7 = y3
    f8 = x2y
    f9 = xy2

    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
