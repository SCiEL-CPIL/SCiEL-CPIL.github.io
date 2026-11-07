# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_basis(x: np.ndarray, y: np.ndarray, degree: int = 4) -> np.ndarray:
    """
    Build a 2D Chebyshev basis matrix of total degree <= `degree`.
    Returns a (n, num_basis) array.
    """
    n = len(x)
    # Precompute Chebyshev polynomials up to `degree` for x and y
    Tx = np.zeros((degree + 1, n))
    Ty = np.zeros((degree + 1, n))
    
    Tx[0, :] = 1.0
    if degree >= 1:
        Tx[1, :] = x
    for k in range(2, degree + 1):
        Tx[k, :] = 2 * x * Tx[k-1, :] - Tx[k-2, :]
        
    Ty[0, :] = 1.0
    if degree >= 1:
        Ty[1, :] = y
    for k in range(2, degree + 1):
        Ty[k, :] = 2 * y * Ty[k-1, :] - Ty[k-2, :]
        
    # Build basis features: T_i(x) * T_j(y) for i+j <= degree
    features = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            features.append(Tx[i, :] * Ty[j, :])
            
    if features:
        return np.column_stack(features)
    else:
        return np.ones((n, 1))


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using a 2D Chebyshev basis.
    The number of basis functions must match len(params).
    We dynamically select the degree to match the parameter count.
    """
    n_params = len(params)
    
    # Determine degree such that number of basis functions <= n_params
    # Number of basis functions for degree d is (d+1)(d+2)/2
    # We want the largest d such that (d+1)(d+2)/2 <= n_params
    # For n_params=10:
    # d=0: 1
    # d=1: 3
    # d=2: 6
    # d=3: 10
    # So d=3 gives exactly 10 basis functions.
    
    # Find appropriate degree
    degree = 0
    while True:
        next_degree = degree + 1
        num_next = (next_degree + 1) * (next_degree + 2) // 2
        if num_next <= n_params:
            degree = next_degree
        else:
            break
            
    # Build basis
    basis = _chebyshev_basis(x, y, degree)
    n_basis = basis.shape[1]
    
    # If we have more params than basis functions, pad with zeros
    # If we have fewer, truncate (should not happen if logic is correct)
    if n_basis < n_params:
        # Pad params with zeros to match basis size? No, we need to match basis to params.
        # Actually, the evaluator passes 10 params. We should use all 10 if possible.
        # The logic above finds max degree. If n_params=10, degree=3, n_basis=10. Perfect.
        # If n_params were 11, degree=3, n_basis=10. We'd ignore the 11th param.
        # This is acceptable as the optimizer will set unused params to whatever, but they won't affect output.
        # However, to be safe and use all parameters if possible, we could add higher order terms?
        # No, let's stick to the standard complete basis up to degree d.
        # If n_params > n_basis, we just use the first n_basis params.
        params_used = params[:n_basis]
    else:
        params_used = params[:n_basis]
        
    return basis @ params_used
# EVOLVE-BLOCK-END
