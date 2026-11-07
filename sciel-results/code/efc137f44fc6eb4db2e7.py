"""Symbolic regression with a fixed rich basis; params are the 10 basis coefficients."""
# EVOLVE-BLOCK-START
import numpy as np


def _basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build the 10-column design matrix for the fixed basis.

    Columns: 1, x, y, x^2, y^2, x*y, sin(x), sin(y), cos(x), cos(y)
    """
    n = x.shape[0]
    B = np.empty((n, 10), dtype=np.float64)
    B[:, 0] = 1.0
    B[:, 1] = x
    B[:, 2] = y
    B[:, 3] = x * x
    B[:, 4] = y * y
    B[:, 5] = x * y
    B[:, 6] = np.sin(x)
    B[:, 7] = np.sin(y)
    B[:, 8] = np.cos(x)
    B[:, 9] = np.cos(y)
    return B


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction = basis(x, y) @ params.

    Linear in params, so the evaluator's BFGS (from ones) reduces to a
    least-squares fit and converges quickly. Unused basis terms get
    coefficient ~0.
    """
    B = _basis(x, y)
    return B @ params
# EVOLVE-BLOCK-END
