"""Symbolic regression with 2 inputs: Chebyshev tensor-product basis + numerical safeguards."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a (n, 10) feature matrix: Chebyshev tensor-product basis up to degree 3.
    
    Uses Chebyshev polynomials T_k which are bounded in [-1,1] on domain [-1,1]
    and provide a well-conditioned basis for BFGS optimization.
    
    Terms (total degree <= 3):
      T0*T0, T1*T0, T0*T1, T2*T0, T0*T2, T1*T1, T3*T0, T0*T3, T2*T1, T1*T2
    
    where T0=1, T1=t, T2=2t^2-1, T3=4t^3-3t
    """
    # Chebyshev polynomials of the first kind
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x

    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y

    f0 = T0_x * T0_y   # 1
    f1 = T1_x * T0_y   # x
    f2 = T0_x * T1_y   # y
    f3 = T2_x * T0_y   # 2x^2 - 1
    f4 = T0_x * T2_y   # 2y^2 - 1
    f5 = T1_x * T1_y   # xy
    f6 = T3_x * T0_y   # 4x^3 - 3x
    f7 = T0_x * T3_y   # 4y^3 - 3y
    f8 = T2_x * T1_y   # (2x^2-1)*y
    f9 = T1_x * T2_y   # x*(2y^2-1)

    F = np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute predictions as linear combination of 10 Chebyshev-basis features.
    
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
