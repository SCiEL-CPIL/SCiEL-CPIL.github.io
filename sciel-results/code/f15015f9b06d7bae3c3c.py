import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = 0.5 * x  # t in [-1, 1]
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1, T_1(t) = t, T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    T = np.zeros((len(x), 10))
    T[:, 0] = 1.0
    T[:, 1] = t
    for n in range(2, 10):
        T[:, n] = 2.0 * t * T[:, n - 1] - T[:, n - 2]
    
    # Linear combination
    return T @ params
