"""
Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
We search for integer/half-integer decompositions of increasing rank.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    def check_exact(U, V, W, rank):
        """
        Check if U, V, W are integers or half-integers and satisfy the identity.
        """
        # Scale by 2 to check for integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        # Check if they are close to integers
        if not np.allclose(U2, np.round(U2), atol=1e-4):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-4):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-4):
            return False
        
        # Convert to exact integers
        U_int = np.round(U2).astype(np.int64)
        V_int = np.round(V2).astype(np.int64)
        W_int = np.round(W2).astype(np.int64)
        
        # Verify identity with integers: sum_r U_int V_int W_int = 4 * delta
        for i in range(n):
            for k in range(p):
                U_slice = U_int[i*m:(i+1)*m, :]
                V_slice = V_int[k*p:(k+1)*p, :]
                w_val = W_int[k*n + i]
                
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                if not np.allclose(mat, 4 * np.eye(m), atol=1e-6):
                    return False
        return True
    
    def compute_residuals(U, V, W, rank):
        """
        Residuals: for all i, k, j1, j2:
        sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] - delta(j1, j2)
        """
        residuals = []
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]  # (m, rank)
                V_slice = V[k*p:(k+1)*p, :]  # (m, rank)
                w_val = W[k*n + i]  # (rank,)
                
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                target = np.eye(m)
                residuals.append(mat.flatten() - target.flatten())
        
        return np.concatenate(residuals)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound for M(n,m,p) is at least max(nm, mp, pn) = max(10, 30, 12) = 30
    min_rank = max(n*m, m*p, p*n)
    
    # We'll try a few optimization strategies
    # 1. Random restarts with gradient descent
    # 2. Random restarts with least squares (if scipy available)
    
    try:
        from scipy.optimize import least_squares
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    def try_rank(rank, time_budget):
        """Try to find a valid decomposition at given rank within time_budget seconds."""
        rank_start = time.time()
        max_restarts = 100
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                return None
            if time.time() - rank_start > time_budget:
                return None
            
            # Random initialization with small values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            if has_scipy:
                def fun(params):
                    len_U = n * m * rank
                    len_V = m * p * rank
                    len_W = p * n * rank
                    
                    U_flat = params[:len_U]
                    V_flat = params[len_U:len_U+len_V]
                    W_flat = params[len_U+len_V:]
                    
                    U = U_flat.reshape((n*m, rank))
                    V = V_flat.reshape((m*p, rank))
                    W = W_flat.reshape((p*n, rank))
                    
                    return compute_residuals(U, V, W, rank)
                
                params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                try:
                    # Use LM method with tighter tolerance
                    result = least_squares(fun, params_init, method='lm', max_nfev=200, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                    
                    len_U = n * m * rank
                    len_V = m * p * rank
                    len_W = p * n * rank
                    
                    U_res = result.x[:len_U].reshape((n*m, rank))
                    V_res = result.x[len_U:len_U+len_V].reshape((m*p, rank))
                    W_res = result.x[len_U+len_V:].reshape((p*n, rank))
                    
                    # Try to snap to half-integers
                    U_snap = np.round(U_res * 2) / 2
                    V_snap = np.round(V_res * 2) / 2
                    W_snap = np.round(W_res * 2) / 2
                    
                    if check_exact(U_snap, V_snap, W_snap, rank):
                        return (U_snap, V_snap, W_snap)
                    
                    # Also try the raw solution
                    if check_exact(U_res, V_res, W_res, rank):
                        return (U_res, V_res, W_res)
                        
                except Exception:
                    pass
            else:
                # Simple gradient descent
                lr = 0.01
                for step in range(200):
                    if time.time() - start_time > DEADLINE - 5:
                        return None
                    if time.time() - rank_start > time_budget:
                        return None
                    
                    residuals = compute_residuals(U, V, W, rank)
                    if np.linalg.norm(residuals) < 1e-4:
                        break
                    
                    # Compute gradients (approximate)
                    eps = 1e-4
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    # This is expensive, so we'll use a simpler approach
                    # Just do a few steps of random perturbation
                    U_new = U + lr * np.random.randn(*U.shape) * 0.1
                    V_new = V + lr * np.random.randn(*V.shape) * 0.1
                    W_new = W + lr * np.random.randn(*W.shape) * 0.1
                    
                    if np.linalg.norm(compute_residuals(U_new, V_new, W_new, rank)) < np.linalg.norm(residuals):
                        U, V, W = U_new, V_new, W_new
                
                if check_exact(U, V, W, rank):
                    return (U, V, W)
        
        return None
    
    # Try ranks from min_rank to best_rank - 1
    # Allocate time budget based on remaining time
    remaining_ranks = best_rank - min_rank
    time_per_rank = (DEADLINE - (time.time() - start_time) - 5) / max(remaining_ranks, 1)
    time_per_rank = max(time_per_rank, 1.0)  # at least 1 second per rank
    
    for rank in range(min_rank, best_rank):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        result = try_rank(rank, time_per_rank)
        if result is not None:
            U_res, V_res, W_res = result
            best_rank = rank
            best_U = U_res
            best_V = V_res
            best_W = W_res
            # Since we're iterating upwards, the first found is minimal
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
