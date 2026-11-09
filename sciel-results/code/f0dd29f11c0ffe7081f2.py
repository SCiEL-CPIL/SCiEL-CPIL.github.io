"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_chebyt

def _chebyshev_basis(x, y, n):
    """
    Build a tensor-product Chebyshev basis of total degree <= n.
    For inputs in [-1, 1], this is a well-conditioned orthogonal-ish basis.
    Returns a matrix of shape (N, num_basis) where N is the number of points.
    """
    # Generate all pairs (i, j) such that i + j <= n
    basis_terms = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            basis_terms.append((i, j))
    
    N = len(x)
    M = len(basis_terms)
    B = np.zeros((N, M))
    
    for k, (i, j) in enumerate(basis_terms):
        cx = eval_chebyt(i, x)
        cy = eval_chebyt(j, y)
        B[:, k] = cx * cy
        
    return B

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Determine the degree based on the number of parameters
    # params has length 10.
    # Total degree n has (n+1)(n+2)/2 terms.
    # n=0: 1 term
    # n=1: 3 terms
    # n=2: 6 terms
    # n=3: 10 terms
    # So if len(params) == 10, we can use degree 3.
    # If len(params) < 10, we use a lower degree.
    # The task says params is an array of length 10.
    
    n_params = len(params)
    
    # Find max degree n such that (n+1)(n+2)/2 <= n_params
    max_degree = 0
    for n in range(10): # Safe upper bound
        num_terms = (n + 1) * (n + 2) // 2
        if num_terms <= n_params:
            max_degree = n
        else:
            break
            
    # Build basis
    B = _chebyshev_basis(x, y, max_degree)
    
    # The basis has M columns. params has n_params elements.
    # If M < n_params, we ignore the extra params (or set them to 0).
    # If M > n_params, we truncate the basis (shouldn't happen with the loop above).
    
    M = B.shape[1]
    
    # Use only the first M params
    active_params = params[:M]
    
    # Compute prediction: B @ active_params
    pred = B @ active_params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(y)
        
    return pred
# EVOLVE-BLOCK-END
