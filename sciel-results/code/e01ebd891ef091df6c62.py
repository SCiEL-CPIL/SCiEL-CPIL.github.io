"""
Structural Change: Optimization-based search for tensor decomposition.
Instead of random sampling or direct construction, we use scipy.optimize
to minimize the residual error of a lower-rank decomposition, initialized
from the schoolbook solution or random values, with a time guard.
"""

import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 5]

def _get_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """
    Compute the residual error for the decomposition.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    We want T[i,j,k] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # Target is all ones
    target = np.ones((n, m, p))
    residual = np.sum((T - target) ** 2)
    return residual

def _try_optimize_rank(rank, n, m, p, time_limit):
    """
    Attempt to find a decomposition of given rank using optimization.
    """
    start_time = time.time()
    
    # Initialize with random values
    np.random.seed(rank) # Deterministic for reproducibility in attempts
    U = np.random.rand(n * m, rank)
    V = np.random.rand(m * p, rank)
    W = np.random.rand(p * n, rank)
    
    # Flatten parameters
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(params):
        U_flat, V_flat, W_flat = params[:n*m*rank], params[n*m*rank:n*m*rank + m*p*rank], params[n*m*rank + m*p*rank:]
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        res = _compute_residual(U, V, W, n, m, p)
        
        # Add a small penalty to encourage sparsity/integers? 
        # For now, just minimize residual.
        return res
    
    # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness better
    # Max iterations and time limit
    max_iter = 500
    result = opt.minimize(objective, params, method='Nelder-Mead', 
                          options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-9})
    
    if time.time() - start_time > time_limit:
        return None
        
    if result.fun < 1e-9:
        # Valid decomposition found
        U_flat, V_flat, W_flat = result.x[:n*m*rank], result.x[n*m*rank:n*m*rank + m*p*rank], result.x[n*m*rank + m*p*rank:]
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        # Convert to list of lists for output
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110 # 10s safety margin
    
    # 1. Get Schoolbook solution (Rank 60)
    U_school, V_school, W_school = _get_schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # 2. Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = 20
    # We try a few ranks below 60.
    # Optimization is slow, so we limit the number of attempts.
    
    ranks_to_try = [50, 40, 30] # Try a few specific ranks
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20: # Keep 20s buffer
            break
            
        # Try optimization for this rank
        # We allow 10 seconds per rank attempt
        res = _try_optimize_rank(rank, n, m, p, time_limit=10)
        
        if res is not None:
            best_result = res
            best_rank = rank
            # If we found a valid one, we could stop, but maybe a lower one exists?
            # Given the difficulty, finding one is a huge win.
            # However, the loop goes 50, 40, 30. If 50 works, 40 might not.
            # We continue to see if we can go lower, but we must be careful with time.
            
    return best_result
