"""Symbolic regression with 2 inputs: Mixed polynomial + trigonometric + rational basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix.
    Strategy: Mixed basis covering polynomial, trigonometric, and rational terms.
    This covers:
    - Constant
    - Linear (x, y)
    - Quadratic (x^2, y^2, xy)
    - Trigonometric (sin(x), sin(y))
    - Rational (1/(1+x^2), 1/(1+y^2))
    This diverse basis can approximate polynomials, oscillatory functions, and rational-like behavior.
    """
    n = x.shape[0]
    
    # Precompute common terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    f0 = np.ones(n)
    f1 = x
    f2 = y
    f3 = x2
    f4 = y2
    f5 = xy
    f6 = np.sin(x)
    f7 = np.sin(y)
    f8 = 1.0 / (1.0 + x2)
    f9 = 1.0 / (1.0 + y2)

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
