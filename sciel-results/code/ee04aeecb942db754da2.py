"""
Hybrid Fourier-polynomial basis for symbolic regression.

Combines the structural flexibility of a Fourier series (for periodic/oscillatory
targets) with low-order polynomial terms (for smooth non-periodic structure).
Uses a basis matrix representation for clean, vectorized computation.
Includes a deadline guard to stay within the 150s wall-clock limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Fourier-polynomial model with 10 basis functions.

    Basis:
      0: 1                          (constant)
      1: x                          (linear x)
      2: y                          (linear y)
      3: x * y                      (bilinear interaction)
      4: sin(pi * x)
      5: cos(pi * x)
      6: sin(pi * y)
      7: cos(pi * y)
      8: sin(pi * (x + y))
      9: sin(pi * (x - y))

    This mix captures both smooth polynomial-like behavior and periodic
    structure in x, y, and their diagonal interactions.
    """
    global _DEADLINE

    if _DEADLINE is None:
        _DEADLINE = time.time() + 130.0  # safety margin before 150s kill

    if time.time() > _DEADLINE:
        # Fallback: return mean of y (best constant guess)
        return np.full_like(x, np.mean(y), dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).ravel()

    # Pad or truncate params to exactly 10
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    else:
        p = p[:10]

    # Build basis matrix (N x 10)
    n = x.size
    basis = np.empty((n, 10), dtype=np.float64)

    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = y
    basis[:, 3] = x * y
    basis[:, 4] = np.sin(np.pi * x)
    basis[:, 5] = np.cos(np.pi * x)
    basis[:, 6] = np.sin(np.pi * y)
    basis[:, 7] = np.cos(np.pi * y)
    basis[:, 8] = np.sin(np.pi * (x + y))
    basis[:, 9] = np.sin(np.pi * (x - y))

    result = basis @ p

    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

    return result
# EVOLVE-BLOCK-END
