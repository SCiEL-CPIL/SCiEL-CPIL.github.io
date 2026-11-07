"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Start time to guard against potential infinite loops or slow operations
    # though Chebyshev evaluation is O(N), this is a safety measure per contract.
    t_start = time.time()
    
    # 10 parameters for a 9th-degree Chebyshev polynomial basis.
    # Chebyshev polynomials are orthogonal on [-1, 1] and provide better
    # numerical conditioning than monomials for approximating smooth functions.
    # We map the input domain [-2, 2] to [-1, 1] for standard Chebyshev definition.
    # t = x / 2.0
    
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # Pre-allocate array for efficiency
    # We need T_0 to T_9, so 10 columns
    T = np.zeros((x.size, 10), dtype=np.float64)
    
    T[:, 0] = 1.0
    T[:, 1] = t
    
    for n in range(2, 10):
        T[:, n] = 2.0 * t * T[:, n-1] - T[:, n-2]
        
        # Safety check: if we are taking too long, break (though unlikely for 10 steps)
        if time.time() - t_start > 110.0:
             # Return current best effort (partial sum) to avoid timeout
             # This is a very unlikely edge case for 10 iterations
             break

    # Linear combination: y = sum(params[i] * T_i(x))
    # Using dot product for vectorized operation
    y = T.dot(params)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
         # If numerical instability occurs, fallback to zero or simple linear
         # But Chebyshev is stable, so this is just a guard
         y = np.where(np.isfinite(y), y, 0.0)
         
    return y
# EVOLVE-BLOCK-END
