import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    # t = x / 2
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    T = np.zeros((10, len(t)), dtype=float)
    T[0, :] = 1.0
    if len(t) > 0:
        T[1, :] = t
        for k in range(2, 10):
            T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Prediction is the linear combination of Chebyshev polynomials
    # y = sum(params[k] * T[k, :])
    # T has shape (10, N), params has shape (10,)
    # We need to sum over the first axis of T, weighted by params
    # This is equivalent to params @ T (resulting in shape (N,))
    y = params @ T
    
    return y
