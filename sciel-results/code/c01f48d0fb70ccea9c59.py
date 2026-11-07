import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Scale input to [-1, 1]
    t = x / 3.0
    
    # Compute Legendre polynomials P_0 to P_9 using recurrence
    # P_0(t) = 1
    # P_1(t) = t
    # (k+1)P_{k+1}(t) = (2k+1)t P_k(t) - k P_{k-1}(t)
    
    # We need a matrix B of shape (N, 10) where B[:, i] = P_i(t)
    n = len(x)
    B = np.zeros((n, 10), dtype=np.float64)
    
    B[:, 0] = 1.0
    if n > 0:
        B[:, 1] = t
        
    for k in range(1, 9):
        # k goes from 1 to 8, calculating P_{k+1}
        # (k+1)P_{k+1} = (2k+1)t P_k - k P_{k-1}
        P_k = B[:, k]
        P_k_minus_1 = B[:, k-1]
        P_k_plus_1 = ((2*k + 1) * t * P_k - k * P_k_minus_1) / (k + 1)
        B[:, k+1] = P_k_plus_1
        
    # result = B @ p
    result = B.dot(p)
    
    return result
