import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    
    # We need a matrix of basis functions B where B[i, j] = T_j(t_i)
    # x has shape (N,)
    # We want result shape (N,)
    
    # Initialize basis matrix
    # N = len(x)
    N = len(x)
    B = np.zeros((N, 10))
    
    # T_0
    B[:, 0] = 1.0
    
    # T_1
    B[:, 1] = t
    
    # T_k for k >= 2
    for k in range(2, 10):
        B[:, k] = 2.0 * t * B[:, k-1] - B[:, k-2]
        
    # Linear combination
    # params has shape (10,)
    # result = B @ params
    return B @ params
