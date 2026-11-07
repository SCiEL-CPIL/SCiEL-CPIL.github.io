"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Legendre polynomial basis of degree 9 on the scaled interval [-1, 1].
    
    x is in [0.1, 6]. We scale to t = (x - 3.05) / 2.95 to map to [-1, 1].
    Then we use Legendre polynomials L_0 through L_9.
    
    Legendre polynomials are orthogonal with respect to the uniform weight function,
    which matches the uniform distribution of the input data. This typically
    provides better numerical conditioning for least-squares (MSE) optimization
    compared to Chebyshev polynomials (which are optimal for minimax error).
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite values
    x = np.where(np.isfinite(x), x, 3.05)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Scale x from [0.1, 6] to [-1, 1]
    # Midpoint = (0.1 + 6) / 2 = 3.05
    # Half-width = (6 - 0.1) / 2 = 2.95
    t = (x - 3.05) / 2.95
    
    # Clip to [-1, 1] to avoid numerical issues (though x should be in [0.1, 6])
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Legendre polynomials L_0 through L_9 using recurrence
    # L_0(t) = 1
    # L_1(t) = t
    # (n+1) L_{n+1}(t) = (2n+1) t L_n(t) - n L_{n-1}(t)
    
    L = np.zeros((10, x.shape[0]))
    L[0, :] = 1.0
    L[1, :] = t
    
    for n in range(1, 9):
        # Calculate L_{n+1}
        # L_{n+1} = ((2n+1)*t*L_n - n*L_{n-1}) / (n+1)
        L[n+1, :] = ((2.0 * n + 1.0) * t * L[n, :] - n * L[n-1, :]) / (n + 1.0)
        
        # Check time budget (safety margin: return current best if close to limit)
        # The evaluation is fast, but we guard against pathological cases or slow hardware
        if time.time() - start_time > 140.0:
             break
    
    # Combine with parameters
    # prediction = sum_{k=0}^{9} p[k] * L_k(t)
    result = np.zeros(x.shape[0])
    for k in range(10):
        result += p[k] * L[k, :]
    
    # Final safety check
    if not np.all(np.isfinite(result)):
        return np.zeros(x.shape[0])
        
    return result
# EVOLVE-BLOCK-END
