"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 145  # safety margin before 150s limit

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _compute_tensor(U, V, W, n, m, p):
    """Compute the decomposed tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T

def _error(T):
    """Compute max absolute error from all-ones tensor."""
    return np.max(np.abs(T - 1.0))

def _smooth_error(T, beta=100.0):
    """
    Smooth surrogate for max|T-1| using log-sum-exp.
    This provides a differentiable proxy for the nonsmooth min/max objective.
    """
    d = np.abs(T - 1.0)
    # Use log-sum-exp for numerical stability
    d_max = np.max(d)
    if d_max < 1e-30:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * d))) / beta
    return log_sum - np.log(d.size) / beta

def _als_iteration(U, V, W, n, m, p):
    """
    One ALS iteration: solve for each factor given the other two.
    Returns new U, V, W.
    """
    rank = U.shape[1]
    U_new = np.zeros((n * m, rank))
    V_new = np.zeros((m * p, rank))
    W_new = np.zeros((p * n, rank))
    
    # Solve for U: for each (i,j), find U[i*m+j,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b_vec = np.ones(p)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                U_new[i * m + j, :] = sol
            except:
                U_new[i * m + j, :] = U[i * m + j, :]
    
    # Solve for V: for each (j,k), find V[j*p+k,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b_vec = np.ones(n)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                V_new[j * p + k, :] = sol
            except:
                V_new[j * p + k, :] = V[j * p + k, :]
    
    # Solve for W: for each (k,i), find W[k*n+i,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b_vec = np.ones(m)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W_new[k * n + i, :] = sol
            except:
                W_new[k * n + i, :] = W[k * n + i, :]
    
    return U_new, V_new, W_new

def _high_precision_polish(U, V, W, n, m, p, max_iter=100, tol=1e-15):
    """
    High-precision polish pass with tighter tolerance and more iterations.
    Uses alternating least squares with careful numerical handling.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(_compute_tensor(U, V, W, n, m, p))
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _als_iteration(U, V, W, n, m, p)
        T = _compute_tensor(U, V, W, n, m, p)
        err = _error(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _als_with_polish(U, V, W, n, m, p, max_iter=80, tol=1e-12):
    """Run ALS iterations with polish and high-precision final pass."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(_compute_tensor(U, V, W, n, m, p))
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _als_iteration(U, V, W, n, m, p)
        T = _compute_tensor(U, V, W, n, m, p)
        err = _error(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Polish pass: tighter tolerance, more iterations
    if best_err < 0.1 and time.time() - _START_TIME < _TIMEOUT - 10:
        U, V, W = best_U, best_V, best_W
        U, V, W, best_err = _high_precision_polish(U, V, W, n, m, p, max_iter=100, tol=1e-15)
    
    return best_U, best_V, best_W, best_err

def _try_rank(target_rank, n, m, p, num_restarts=8):
    """Try to find a rank-target_rank decomposition with multiple initialization strategies."""
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        # Different initialization strategies for different restarts
        if restart == 0:
            # Structured init: partial schoolbook
            np.random.seed(42 + restart * 1000)
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            for r in range(target_rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                U[i * m + j, r] += 0.01 * np.random.randn()
                V[j * p + k, r] += 0.01 * np.random.randn()
                W[k * n + i, r] += 0.01 * np.random.randn()
        elif restart == 1:
            # Random with normalization
            np.random.seed(42 + restart * 1000)
            U = np.random.randn(n * m, target_rank) / np.sqrt(target_rank)
            V = np.random.randn(m * p, target_rank) / np.sqrt(target_rank)
            W = np.random.randn(p * n, target_rank) / np.sqrt(target_rank)
        elif restart == 2:
            # Sparse-like initialization
            np.random.seed(42 + restart * 1000)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            # Zero out some entries to create sparsity
            U[np.random.random(U.shape) < 0.5] = 0
            V[np.random.random(V.shape) < 0.5] = 0
            W[np.random.random(W.shape) < 0.5] = 0
            U /= (np.sqrt(target_rank) + 1e-8)
            V /= (np.sqrt(target_rank) + 1e-8)
            W /= (np.sqrt(target_rank) + 1e-8)
        else:
            # Pure random
            np.random.seed(42 + restart * 1000)
            U = np.random.randn(n * m, target_rank) / np.sqrt(target_rank)
            V = np.random.randn(m * p, target_rank) / np.sqrt(target_rank)
            W = np.random.randn(p * n, target_rank) / np.sqrt(target_rank)
        
        U_sol, V_sol, W_sol, err = _als_with_polish(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = max_rank
    
    # Try to find lower rank decompositions
    # Known lower bound: for 3x4 * 4x4, rank is at least 33
    # Try ranks from 33 up to 47, but prioritize lower ranks with more restarts
    for target_rank in range(33, max_rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        
        # Use more restarts for lower ranks where we're more likely to succeed
        if target_rank <= 35:
            num_restarts = 10
        elif target_rank <= 40:
            num_restarts = 8
        else:
            num_restarts = 5
        
        U_try, V_try, W_try, err = _try_rank(target_rank, n, m, p, num_restarts=num_restarts)
        
        if err < 1e-10:
            # Found exact decomposition
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = target_rank
            # Try even lower
            continue
        else:
            # If error is small, we might be close - try more restarts
            if err < 0.01 and time.time() - _START_TIME < _TIMEOUT - 10:
                U_try2, V_try2, W_try2, err2 = _try_rank(target_rank, n, m, p, num_restarts=10)
                if err2 < 1e-10:
                    best_U, best_V, best_W = U_try2, V_try2, W_try2
                    best_rank = target_rank
                    continue
    
    # Convert to lists for return
    U_list = [[float(U_list_val) for U_list_val in row] for row in best_U.tolist()]
    V_list = [[float(V_list_val) for V_list_val in row] for row in best_V.tolist()]
    W_list = [[float(W_list_val) for W_list_val in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
