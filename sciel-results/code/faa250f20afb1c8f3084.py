"""Program to find a low-rank exact decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0  # Reduced margin to allow more work
    
    n, m, p = 3, 3, 5
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (15, rank)
    # W shape: (p*n, rank) = (15, rank)
    
    def schoolbook_decomposition():
        """Return rank-45 schoolbook decomposition."""
        rank = n * m * p  # 45
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact using vectorized operations."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        max_err = np.max(np.abs(T - 1.0))
        return max_err < tol, max_err
    
    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integers."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def polish_with_optimizer(U, V, W, max_time=10.0):
        """
        Use scipy.optimize.minimize to polish the decomposition.
        Minimizes sum of squared errors.
        """
        start_time = time.time()
        
        def objective(params):
            if time.time() > start_time + max_time:
                return 1e10
            U_flat, V_flat, W_flat = np.split(params, [U.size, U.size + V.size])
            U_p = U_flat.reshape(U.shape)
            V_p = V_flat.reshape(V.shape)
            W_p = W_flat.reshape(W.shape)
            _, err = check_decomposition(U_p, V_p, W_p, tol=1e15)
            return err**2
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B for better convergence on large problems
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15})
        
        if res.success or res.fun < 1e-10:
            U_opt, V_opt, W_opt = np.split(res.x, [U.size, U.size + V.size])
            return U_opt.reshape(U.shape), V_opt.reshape(V.shape), W_opt.reshape(W.shape)
        
        return U, V, W
    
    def polish_decomposition(U, V, W, max_iters=50, tol=1e-10):
        """
        High-precision polish pass using alternating least squares with tight tolerance.
        """
        for it in range(max_iters):
            if time.time() > deadline - margin:
                break
            
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, U.shape[1]))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, V.shape[1]))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, W.shape[1]))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            
            U, V, W = U_new, V_new, W_new
            
            is_valid, _ = check_decomposition(U, V, W, tol=tol)
            if is_valid:
                return U, V, W
        
        return U, V, W
    
    def alternating_minimization(rank, max_iters=300, tol=1e-8, seed=42):
        """
        Alternating least squares for tensor decomposition.
        Fix two factors, solve for the third via linear least squares.
        """
        np.random.seed(seed)
        
        # Initialize with random values, scaled to avoid huge numbers
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize columns to prevent scaling issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10:
                U[:, r] /= norm_u
            if norm_v > 1e-10:
                V[:, r] /= norm_v
            if norm_w > 1e-10:
                W[:, r] /= norm_w
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - margin:
                break
            
            # Solve for U: fix V and W
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            
            # Solve for V: fix U and W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            
            # Solve for W: fix U and V
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            is_valid, err = check_decomposition(U, V, W, tol=1e-4)
            if is_valid:
                # Try rounding
                U_r, V_r, W_r = round_to_half_integers(U, V, W)
                is_valid_r, _ = check_decomposition(U_r, V_r, W_r, tol=1e-4)
                if is_valid_r:
                    return U_r, V_r, W_r, it
                else:
                    return U, V, W, it
            
            if abs(err - prev_err) < 1e-12:
                break
            prev_err = err
        
        # If close, try to polish with optimizer
        is_close, err = check_decomposition(U, V, W, tol=1e-2)
        if is_close:
            U_p, V_p, W_p = polish_with_optimizer(U, V, W, max_time=5.0)
            is_valid_p, _ = check_decomposition(U_p, V_p, W_p, tol=1e-8)
            if is_valid_p:
                U_r, V_r, W_r = round_to_half_integers(U_p, V_p, W_p)
                is_valid_r, _ = check_decomposition(U_r, V_r, W_r, tol=1e-4)
                if is_valid_r:
                    return U_r, V_r, W_r, max_iters
                else:
                    return U_p, V_p, W_p, max_iters
        
        return None, None, None, max_iters
    
    # Try to find low-rank decompositions
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 45
    
    # Try ranks from 44 down to 30
    # We prioritize higher ranks first as they are easier to find, then try to go lower
    for rank in range(44, 29, -1):
        if time.time() > deadline - margin:
            break
        
        # Try multiple random seeds
        found = False
        num_seeds = 50 if rank > 40 else 100
        for seed in range(num_seeds):
            if time.time() > deadline - margin:
                break
            
            U_try, V_try, W_try, iters = alternating_minimization(rank, max_iters=200, seed=seed)
            if U_try is not None:
                is_valid, _ = check_decomposition(U_try, V_try, W_try, tol=1e-4)
                if is_valid:
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # If we found a solution, we can try to see if we can go lower immediately?
            # The loop continues to lower ranks automatically.
            pass
    
    # If we found something, convert to list of lists
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: Schoolbook
    U, V, W = schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
