"""
Symbolic regression using a hybrid Chebyshev + RBF basis.
The hidden function is approximated by a linear combination of Chebyshev polynomials
and Radial Basis Functions (Gaussians) to capture both global trends and local features.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values
    """
    # Deadline guard: if we are exceeding time, return a simple fallback
    # (In this linear model, computation is fast, but this satisfies the requirement)
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    # Scale x from [-2, 2] to [-1, 1] for Chebyshev
    t = 0.5 * x
    
    n_params = 10
    basis = np.zeros((len(x), n_params))
    
    # First 5 parameters: Chebyshev polynomials T_0 to T_4
    n_cheb = 5
    if n_cheb > 0:
        basis[:, 0] = 1.0
    if n_cheb > 1:
        basis[:, 1] = t
        
    for k in range(2, n_cheb):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Remaining 5 parameters: Gaussian RBFs
    n_rbf = n_params - n_cheb
    # Centers spread across [-1.5, 1.5] in the scaled domain t
    # t is in [-1, 1], but we can place centers slightly outside or inside
    # Let's use centers at -1, -0.5, 0, 0.5, 1
    centers = np.linspace(-1.0, 1.0, n_rbf)
    sigma = 0.5 # Width of the Gaussians
    
    for i in range(n_rbf):
        idx = n_cheb + i
        basis[:, idx] = np.exp(-((t - centers[i])**2) / (2 * sigma**2))
        
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
