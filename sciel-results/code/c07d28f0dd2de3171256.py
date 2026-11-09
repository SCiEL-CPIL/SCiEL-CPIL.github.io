"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis.
    
    Chebyshev polynomials are orthogonal on [-1, 1] and provide a 
    well-conditioned basis for approximating smooth functions on 
    a bounded interval. We map x from [-2, 2] to t in [-1, 1] 
    and use the first 10 Chebyshev polynomials T_0(t) to T_9(t).
    
    This linear model is robust to the BFGS optimizer starting 
    from all-ones, as the loss landscape is convex.
    """
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_n(t) using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n}(t) = 2t T_{n-1}(t) - T_{n-2}(t)
    
    # We need 10 basis functions
    T_prev2 = np.ones_like(t)  # T_0
    T_prev1 = t.copy()          # T_1
    
    # Store all T_n in a list or array for dot product
    # To be efficient, we can accumulate the result directly
    # result = p0 * T_0 + p1 * T_1 + ... + p9 * T_9
    
    result = params[0] * T_prev2 + params[1] * T_prev1
    
    T_curr = T_prev1
    T_prev = T_prev2
    
    for n in range(2, 10):
        # T_n = 2*t*T_{n-1} - T_{n-2}
        T_curr = 2 * t * T_prev - T_prev2
        
        # Add to result
        result += params[n] * T_curr
        
        # Update pointers
        T_prev2 = T_prev
        T_prev = T_curr
    
    return result
# EVOLVE-BLOCK-END
