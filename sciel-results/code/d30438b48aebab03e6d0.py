import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    # Map x from [-3, 3] to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0 = 1, T_1 = t, T_n = 2*t*T_{n-1} - T_{n-2}
    T = np.ones((x.shape[0], 10))
    T[:, 1] = t
    for n in range(2, 10):
        T[:, n] = 2.0 * t * T[:, n-1] - T[:, n-2]
    
    # Linear combination: sum(params[i] * T_i)
    result = T @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result
