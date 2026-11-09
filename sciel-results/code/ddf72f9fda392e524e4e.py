"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-3, 3] to [-1, 1] for Chebyshev basis
    t = x / 3.0
    
    # Build Chebyshev polynomial basis T_0(t) through T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    n_basis = 10
    basis = np.zeros((len(x), n_basis))
    basis[:, 0] = 1.0
    if n_basis > 1:
        basis[:, 1] = t
    for k in range(2, n_basis):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    
    # Linear combination of Chebyshev basis functions with params as coefficients
    return basis @ params
# EVOLVE-BLOCK-END
