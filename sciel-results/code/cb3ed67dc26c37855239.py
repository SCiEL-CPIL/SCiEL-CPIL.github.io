"""Symbolic regression with 2 inputs: complete cubic polynomial basis + numerical safeguards."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a (n, 10) feature matrix: complete homogeneous cubic polynomial in x,y.
    
    Terms:
      1, x, y, x^2, y^2, xy, x^3, y^3, x^2*y, x*y^2
    
    This spans the full space of polynomials of total degree <= 3,
    which is the maximal expressiveness for 10 parameters with smooth
    polynomial structure.
    """
    f0 = np.ones_like(x)
    f1 = x
    f2 = y
    f3 = x * x
    f4 = y * y
    f5 = x * y
    f6 = x * x * x
    f7 = y * y * y
    f8 = x * x * y
    f9 = x * y * y

    F = np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute predictions as linear combination of 10 polynomial features.
    
    The evaluator fits params via BFGS to minimize training MSE.
    This function must be fast and numerically stable.
    """
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        pred = F @ params
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred

    F = _features(x, y)

    # Compute prediction
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
