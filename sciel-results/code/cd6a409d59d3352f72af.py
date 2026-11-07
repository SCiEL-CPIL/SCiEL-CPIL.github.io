"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid basis of Chebyshev polynomials
    and Gaussian RBFs, linear in the 10 parameters.
    
    Strategy:
    - 8 Chebyshev polynomials T_k(x/3) for k=0..7 to capture smooth global trends.
    - 2 Gaussian RBFs centered at -1 and 1 with width 1.5 to capture local features.
    This provides a rich, non-linear, yet linear-in-params basis that BFGS can
    efficiently fit to a wide variety of target functions.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Normalize x to [-1, 1] for Chebyshev stability, since x is in [-3, 3]
    x_norm = x / 3.0
    
    # Build basis matrix
    # Terms 0-7: Chebyshev polynomials T_0 to T_7
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)
    T = np.zeros((8, len(x_norm)))
    T[0, :] = 1.0
    if len(x_norm) > 0:
        T[1, :] = x_norm
    for k in range(2, 8):
        T[k, :] = 2 * x_norm * T[k-1, :] - T[k-2, :]
    
    # Terms 8-9: Gaussian RBFs
    # Centered at -1 and 1 in original scale, width sigma=1.5
    sigma = 1.5
    rbf1 = np.exp(-0.5 * ((x - 1.0) / sigma) ** 2)
    rbf2 = np.exp(-0.5 * ((x + 1.0) / sigma) ** 2)
    
    # Stack basis functions
    basis = np.vstack([
        T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7],
        rbf1, rbf2
    ])
    
    # Linear combination: y = params @ basis
    # params is shape (10,), basis is shape (10, N)
    # result is shape (N,)
    return basis.T @ params

# EVOLVE-BLOCK-END
