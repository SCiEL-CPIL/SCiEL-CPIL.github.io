"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a well-conditioned basis of 10 terms.
    
    The basis uses Chebyshev polynomials T_0 to T_9 on the mapped domain [-1, 1].
    Chebyshev polynomials are orthogonal on [-1, 1], providing a well-conditioned
    basis for least-squares fitting. This allows BFGS to converge reliably to
    the optimal polynomial approximation of the target function.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for stability
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute Chebyshev polynomials T_0 to T_9 using recurrence
    # T_0(t) = 1, T_1(t) = t, T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    T = np.empty((10, len(t_safe)), dtype=np.float64)
    T[0] = np.ones_like(t_safe)
    T[1] = t_safe.copy()
    for n in range(2, 10):
        T[n] = 2.0 * t_safe * T[n-1] - T[n-2]
    
    # Build the linear combination: sum_{k=0}^{9} params[k] * T_k(t)
    result = np.zeros_like(t_safe)
    for k in range(10):
        result += params[k] * T[k]
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
