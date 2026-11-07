import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Number of parameters is 10, so we use degrees 0 to 9
    n_params = len(params)
    
    # Initialize basis matrix
    # basis[i, k] = T_k(t_i)
    basis = np.zeros((len(x), n_params))
    
    if n_params > 0:
        basis[:, 0] = 1.0
    if n_params > 1:
        basis[:, 1] = t
        
    for k in range(2, n_params):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Compute prediction: sum(params[k] * basis[:, k])
    # params is shape (10,), basis is shape (N, 10)
    # result is shape (N,)
    return basis @ params
