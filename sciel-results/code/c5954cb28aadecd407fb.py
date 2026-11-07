"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using 10 Gaussian Radial Basis Functions (RBFs)
    centered at distinct points in [-3, 3] with varying widths.
    
    The 10 parameters independently adjust the amplitude of each RBF, allowing
    the model to capture arbitrary non-linear features that global polynomials
    or Fourier series may underfit.
    
    Centers are evenly spaced across [-3, 3] and widths are chosen to provide
    a mix of narrow (local detail) and wide (global trend) basis functions.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)

    # 10 Gaussian RBFs with distinct centers in [-3, 3]
    # Centers evenly spaced
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Varying widths: mix of narrow and wide for local and global features
    # Pattern: narrow, medium, narrow, wide, narrow, wide, narrow, medium, narrow, medium
    widths = np.array([0.5, 0.8, 0.5, 1.0, 0.5, 1.5, 0.5, 1.0, 0.8, 0.5])
    
    # Compute RBFs: exp(-((x - c) / w)^2)
    # x shape: (n,), centers shape: (10,), widths shape: (10,)
    # We want basis shape: (n, 10) where basis[i, j] = RBF(x_i, c_j, w_j)
    
    # Efficient computation using broadcasting
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    scaled = diff / widths[np.newaxis, :]              # (n, 10)
    basis = np.exp(-scaled**2)                         # (n, 10)
    
    # Linear combination of RBFs
    result = basis @ params  # (n,)
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    # Deadline check (computation is fast, this is just a safety net)
    if time.time() > deadline:
        pass  # already computed, just return
    
    return result
# EVOLVE-BLOCK-END
