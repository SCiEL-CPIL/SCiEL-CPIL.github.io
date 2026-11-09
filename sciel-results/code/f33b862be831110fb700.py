"""Symbolic regression with 2 inputs: stronger basis + numerical safeguards."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a (n, 10) feature matrix covering common smooth functions."""
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
    # Radial
    f6 = r2
    # Cubic
    f7 = x * x2        # x^3
    f8 = y * y2        # y^3
    # Interaction cubic
    f9 = xy * (x + y)  # x^2*y + x*y^2

    F = np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)

    # Ensure finite output
    pred = F @ params
    if not np.all(np.isfinite(pred)):
        # Fallback: zero out non-finite contributions
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
