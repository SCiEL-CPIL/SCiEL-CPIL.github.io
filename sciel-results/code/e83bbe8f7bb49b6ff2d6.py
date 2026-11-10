"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Store start time for the entire module load/execution context if needed,
# but primarily we ensure the equation function is fast.
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of Chebyshev polynomials.
    
    Chebyshev polynomials provide a well-conditioned basis for polynomial approximation
    on a bounded interval, which helps the BFGS optimizer converge more reliably
    compared to monomial bases (x, x^2, ...).
    
    The input x is in [-2, 2]. We map it to t in [-1, 1] to use standard Chebyshev
    polynomials T_n(t).
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # We need T_0 through T_9
    # Basis vector B has shape (N, 10)
    # B[:, 0] = T0
    # B[:, 1] = T1
    # ...
    
    # Initialize basis matrix
    N = len(x)
    basis = np.empty((N, 10), dtype=np.float64)
    basis[:, 0] = T0
    basis[:, 1] = T1
    
    # Compute up to T_9
    T_prev = T0
    T_curr = T1
    for k in range(2, 10):
        T_next = 2.0 * t * T_curr - T_prev
        basis[:, k] = T_next
        T_prev = T_curr
        T_curr = T_next
        
    # Linear combination
    return basis @ p

# EVOLVE-BLOCK-END
