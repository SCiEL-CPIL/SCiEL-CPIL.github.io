"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """Compute Chebyshev polynomials T_0(z), T_1(z), ..., T_degree(z).
    
    Uses the recurrence: T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    
    Returns array of shape (N, degree+1).
    """
    N = z.shape[0]
    basis = np.empty((N, degree + 1), dtype=np.float64)
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = z
        for k in range(2, degree + 1):
            basis[:, k] = 2.0 * z * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize input from [-2, 2] to [-1, 1] for stable Chebyshev evaluation
    z = x / 2.0
    
    # Build Chebyshev basis T_0(z), T_1(z), ..., T_9(z) — 10 basis functions
    basis = _chebyshev_basis(z, 9)
    
    # Linear combination of Chebyshev basis with 10 parameters
    # basis shape: (N, 10), params shape: (10,)
    return basis @ params
# EVOLVE-BLOCK-END
