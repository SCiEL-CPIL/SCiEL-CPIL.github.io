"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_rbf_basis(x: np.ndarray) -> np.ndarray:
    """
    Build an RBF basis matrix with fixed centers and bandwidths.
    Returns shape (n_points, 10) where columns are:
      - 1 constant (for offset)
      - 9 Gaussian RBF kernels at fixed centers
    Centers are spread across [0.1, 6] with varying bandwidths
    to capture both global trends and local features.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Fixed centers spread across the domain [0.1, 6]
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 1.0, 3.0, 5.0])
    
    # Fixed bandwidths: mix of narrow (local) and wide (global)
    bandwidths = np.array([0.5, 0.8, 1.0, 1.2, 0.6, 0.9, 1.5, 0.7, 1.1])
    
    # Compute RBF values: exp(-((x - c)^2) / (2 * sigma^2))
    # Shape: (n_points, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 9)
    sigma2 = bandwidths[np.newaxis, :] ** 2  # (1, 9)
    rbf = np.exp(-0.5 * (diff ** 2) / sigma2)  # (n, 9)
    
    # Add constant column
    const_col = np.ones((n, 1))
    basis = np.hstack([const_col, rbf])  # (n, 10)
    
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based symbolic regression: linear combination of Gaussian RBF kernels
    at fixed centers plus a constant term. This provides high local flexibility
    while maintaining linear-in-parameters structure for BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Build basis matrix (n_points, 10)
    basis = _get_rbf_basis(x)
    
    # Linear combination: y = basis @ params
    y = basis @ params
    
    return y

# EVOLVE-BLOCK-END
