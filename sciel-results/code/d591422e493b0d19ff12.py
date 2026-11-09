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

def _try_optimize_rank(rank, n, m, p, time_limit, seed_offset=0):
    """
    Attempt to find a decomposition of given rank using optimization.
    """
    start_time = time.time()
    
    # We will try multiple initializations
    num_restarts = 5
    best_result = None
    best_fun = np.inf
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random values
        np.random.seed(rank * 100 + seed_offset + restart)
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
            return res
        
        # Use Nelder-Mead
        # Reduce maxiter if time is tight
        time_remaining = time_limit - (time.time() - start_time)
        max_iter = int(200 * (time_remaining / time_limit))
        if max_iter < 50:
            max_iter = 50
            
        try:
            result = opt.minimize(objective, params, method='Nelder-Mead', 
                                  options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-8})
        except Exception:
            continue
            
        if result.fun < best_fun:
            best_fun = result.fun
            if result.fun < 1e-9:
                U_flat, V_flat, W_flat = result.x[:n*m*rank], result.x[n*m*rank:n*m*rank + m*p*rank], result.x[n*m*rank + m*p*rank:]
                U = U_flat.reshape(n * m, rank)
                V = V_flat.reshape(m * p, rank)
                W = W_flat.reshape(p * n, rank)
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # Found a valid one, can return early if we want, but maybe check time?
                # For now, return immediately as finding one is rare and valuable
                return best_result
                
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130 # 20s safety margin
    
    # 1. Get Schoolbook solution (Rank 60)
    U_school, V_school, W_school = _get_schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # 2. Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = 20
    # We try a few ranks below 60.
    # Optimization is slow, so we limit the number of attempts.
    
    # We try ranks in increasing order of difficulty (lower rank is harder).
    # But usually, we want to find the *lowest* possible rank.
    # However, optimization is local.
    # Let's try a few specific ranks.
    
    ranks_to_try = [25, 30, 35, 40, 45, 50]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15: # Keep 15s buffer
            break
            
        # Try optimization for this rank
        # We allow a bit more time for lower ranks? Or equal time?
        # Let's give 10s per rank attempt.
        res = _try_optimize_rank(rank, n, m, p, time_limit=10)
        
        if res is not None:
            # Since we are iterating from low to high, the first one found is the best (lowest rank)
            # among the successful ones.
            best_result = res
            best_rank = rank
            break # Found the lowest rank in our list, stop.
            
    return best_result
