import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for standard Legendre polynomials
    t = x / 2.0
    
    # Pre-allocate space for 10 basis functions
    # Shape: (10, N) where N is number of points
    N = len(x)
    P = np.zeros((10, N))
    
    P[0, :] = 1.0
    P[1, :] = t
    
    # Recurrence relation for Legendre polynomials
    # k * P_k(t) = (2k - 1) * t * P_{k-1}(t) - (k - 1) * P_{k-2}(t)
    for k in range(2, 10):
        P[k, :] = ((2 * k - 1) * t * P[k-1, :] - (k - 1) * P[k-2, :]) / k
        
    # Compute linear combination
    # params is shape (10,), P is shape (10, N)
    # We want shape (N,)
    y = np.dot(params, P)
    
    return y
