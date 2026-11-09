"""Symbolic regression with 2 inputs: richer basis + numerical safeguards."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a (n, 10) feature matrix covering common smooth functions.
    
    Weights are chosen to cover:
    - Constant, linear, quadratic terms (full polynomial up to degree 2)
    - Key cubic terms for better approximation
    - Radial and interaction terms for non-polynomial structure
    """
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2

    # Constant
    f0 = np.ones_like(x)
    # Linear
    f1 = x
    f2 = y
    # Quadratic
    f3 = x2
    f4 = y2
    f5 = xy
    # Radial quadratic
    f6 = r2
    # Cubic terms
    f7 = x * x2        # x^3
    f8 = y * y2        # y^3
    # Mixed cubic: x^2*y + x*y^2 = xy(x+y)
    f9 = xy * (x + y)

    F = np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute predictions as linear combination of 10 basis features.
    
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
        mask = np.isfinite(pred)
        pred = np.where(mask, pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
