"""Symbolic regression with a hybrid Chebyshev + RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 110.0  # 10s margin before 120s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of 5 Chebyshev polynomials and 5 RBFs.
    Input x in [-3, 3].
    Basis:
    - T_0(t) to T_4(t) with t = x/3
    - 5 RBFs with fixed centers at -3, -1.5, 0, 1.5, 3 and gamma=1.0
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a safe default if we are about to time out
        # This shouldn't happen during normal fitting steps which are fast,
        # but protects against pathological cases.
        return np.zeros_like(x)

    n = len(x)
    if n == 0:
        return np.array([])

    # Map x to t in [-1, 1] for Chebyshev
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0(t) through T_4(t)
    # We need 5 Chebyshev terms (indices 0-4)
    # b_cheb shape: (n, 5)
    b_cheb = np.zeros((n, 5))
    b_cheb[:, 0] = 1.0
    if n > 0:
        b_cheb[:, 1] = t
        for k in range(2, 5):
            b_cheb[:, k] = 2.0 * t * b_cheb[:, k-1] - b_cheb[:, k-2]
    
    # Compute RBFs
    # Centers: -3, -1.5, 0, 1.5, 3
    # Gamma: 1.0
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    gamma = 1.0
    
    # b_rbf shape: (n, 5)
    # x is (n, 1) effectively, centers is (5,)
    # diff = x[:, None] - centers[None, :]
    diff = x[:, None] - centers[None, :]
    b_rbf = np.exp(-gamma * diff**2)
    
    # Combine bases: (n, 10)
    basis = np.hstack((b_cheb, b_rbf))
    
    # Linear combination: y = basis @ params
    # basis: (n, 10), params: (10,) -> result: (n,)
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
