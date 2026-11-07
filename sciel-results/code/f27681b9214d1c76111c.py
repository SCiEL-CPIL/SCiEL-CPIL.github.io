# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() > _DEADLINE:
        p = np.asarray(params, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        if p.shape[0] >= 2:
            return p[0] * x_arr + p[1]
        return np.zeros_like(x_arr)
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
    
    # Strategy: Chebyshev polynomial expansion of degree 9 on [-3, 3]
    # Map x from [-3, 3] to t in [-1, 1] for standard Chebyshev polynomials
    # t = x / 3
    # f(x) = sum_{k=0}^{9} p[k] * T_k(x/3)
    #
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    #
    # This is a complete basis for smooth functions on [-3,3] and is
    # numerically stable (no monomial explosion). BFGS fits all 10
    # coefficients in a convex-like landscape (linear in params).
    
    t = x / 3.0  # map to [-1, 1]
    
    # Compute Chebyshev polynomials T_0 through T_9
    T = np.empty((10, x.shape[0]), dtype=np.float64)
    T[0] = 1.0
    T[1] = t
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]
    
    # result = sum_k p[k] * T_k
    # Use matrix-vector product: T is (10, N), p is (10,)
    result = T @ p[:10]
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback: simple linear
        return p[0] * x + p[1]
    
    return result
# EVOLVE-BLOCK-END
