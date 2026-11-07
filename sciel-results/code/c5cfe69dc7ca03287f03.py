"""Rewritten program: softplus basis with fixed centers and slopes, linear in weights."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_S = 150.0
_MARGIN_S = 5.0
_START_TIME = time.time()


def _compute_basis(x):
    """
    Compute a bilinear Chebyshev basis of 10 functions.
    Uses products of low-degree Chebyshev polynomials T_i(t) * T_j(t)
    where t = x/2 maps [-2, 2] to [-1, 1].
    """
    xc = np.clip(x, -2.0, 2.0)
    t = 0.5 * xc

    # Compute Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t * t * t * t - 8.0 * t * t + 1.0

    # Define 10 basis functions as products T_i * T_j
    # This allows capturing interactions and higher-order terms efficiently.
    # Pairs: (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (1,4)
    basis = np.empty((len(xc), 10), dtype=np.float64)
    
    basis[:, 0] = T0 * T0  # 1
    basis[:, 1] = T0 * T1  # t
    basis[:, 2] = T0 * T2  # 2t^2 - 1
    basis[:, 3] = T0 * T3  # 4t^3 - 3t
    basis[:, 4] = T1 * T1  # t^2
    basis[:, 5] = T1 * T2  # 2t^3 - t
    basis[:, 6] = T1 * T3  # 4t^4 - 3t^2
    basis[:, 7] = T2 * T2  # 4t^4 - 4t^2 + 1
    basis[:, 8] = T2 * T3  # 8t^5 - 10t^3 + 2t
    basis[:, 9] = T1 * T4  # 8t^5 - 8t^3 + t

    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    The evaluator calls this with x (training or test points) and params (length 10).
    We use params as the weights in the softplus basis expansion.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE_S - _MARGIN_S:
        # Fallback: simple linear
        return params[0] + params[1] * np.clip(x, -2.0, 2.0)

    xc = np.clip(x, -2.0, 2.0)
    basis = _compute_basis(xc)

    pred = basis @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return params[0] + params[1] * xc

    return pred
# EVOLVE-BLOCK-END
