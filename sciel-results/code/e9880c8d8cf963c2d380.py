"""Symbolic regression with 2 inputs: Rich diverse basis (polynomials, trig, tanh) with normalization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix using a diverse basis of nonlinear terms.
    
    Mix of polynomial, trigonometric, and tanh terms in tensor-product form
    to give BFGS a rich search space over diverse function families.
    """
    n = x.shape[0]

    # Basic x features
    x1 = x
    x2 = x * x
    x3 = x * x * x
    sx = np.sin(np.pi * x)
    cx = np.cos(np.pi * x)
    tx = np.tanh(x)

    # Basic y features
    y1 = y
    y2 = y * y
    y3 = y * y * y
    sy = np.sin(np.pi * y)
    cy = np.cos(np.pi * y)
    ty = np.tanh(y)

    # 10 diverse features: mix of linear, quadratic, trig, tanh, and cross terms
    f0 = np.ones(n)                          # constant
    f1 = x1                                  # linear x
    f2 = y1                                  # linear y
    f3 = x2                                  # quadratic x
    f4 = y2                                  # quadratic y
    f5 = x1 * y1                             # interaction x*y
    f6 = sx * sy                             # sin(pi*x)*sin(pi*y)
    f7 = cx * cy                             # cos(pi*x)*cos(pi*y)
    f8 = tx * ty                             # tanh(x)*tanh(y)
    f9 = x3 + y3                             # cubic (sum to keep 10 features)

    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # Center and scale each feature for better conditioning
    # Use running stats to avoid storing full arrays
    for j in range(10):
        col = F[:, j]
        mean_j = np.mean(col)
        std_j = np.std(col)
        if std_j > 1e-12:
            F[:, j] = (col - mean_j) / std_j
        else:
            F[:, j] = col - mean_j

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
