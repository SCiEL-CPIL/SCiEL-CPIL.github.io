"""Symbolic regression with 2 inputs: Chebyshev tensor-product basis (degree 3) with normalization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix using a normalized Chebyshev tensor-product basis.
    
    The basis consists of products $T_i(x) T_j(y)$ where $i, j \ge 0$ and $i+j \le 3$.
    To improve the conditioning for BFGS optimization, non-constant Chebyshev terms
    are scaled by sqrt(2) to have unit variance under uniform distribution on [-1, 1].
    """
    n = x.shape[0]

    # Compute Chebyshev polynomials for x and y
    x2 = x * x
    y2 = y * y
    
    Tx0 = np.ones(n)
    Tx1 = x
    Tx2 = 2.0 * x2 - 1.0
    Tx3 = 4.0 * x2 * x - 3.0 * x

    Ty0 = np.ones(n)
    Ty1 = y
    Ty2 = 2.0 * y2 - 1.0
    Ty3 = 4.0 * y2 * y - 3.0 * y

    # Scale non-constant terms to have unit variance (approx)
    # E[T_k^2] = 1 for k=0, 0.5 for k>0. Scale by sqrt(2) for k>0.
    S = np.sqrt(2.0)
    
    f0 = Tx0 * Ty0  # 1
    f1 = S * Tx1 * Ty0  # sqrt(2) x
    f2 = S * Tx0 * Ty1  # sqrt(2) y
    f3 = S * Tx2 * Ty0  # sqrt(2) (2x^2 - 1)
    f4 = S * S * Tx1 * Ty1  # 2 xy (variance 0.5*0.5*4 = 1)
    f5 = S * Tx0 * Ty2  # sqrt(2) (2y^2 - 1)
    f6 = S * Tx3 * Ty0  # sqrt(2) (4x^3 - 3x)
    f7 = S * S * Tx2 * Ty1  # 2 (2x^2 - 1)y
    f8 = S * S * Tx1 * Ty2  # 2 x(2y^2 - 1)
    f9 = S * Tx0 * Ty3  # sqrt(2) (4y^3 - 3y)

    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        pred = F @ params
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
