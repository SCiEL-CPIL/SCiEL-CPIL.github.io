"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T[i,j,k] for A (n x m) * B (m x p) = C (n x p).
    The tensor T has entries T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    for some i, j, k. All other entries are 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_exact(T, U, V, W, tol=1e-6):
    """
    Verify that the decomposition is exact (within tolerance).
    """
    r = U.shape[1]
    recon = np.zeros(T.shape, dtype=np.float64)
    for rr in range(r):
        recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    residual = T - recon
    return np.max(np.abs(residual)) < tol

def _try_rank_als(T, n, m, p, rank, time_limit, seed=0):
    """
    Try to find a rank-`rank` decomposition using ALS (Alternating Least Squares).
    Returns (U, V, W) if found, else None.
    """
    np.random.seed(seed)
    
    a_size = n * m
    b_size = m * p
    c_size = p * n
    
    # Precompute the non-zero entries of T for efficient residual computation
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    nz_a, nz_b, nz_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_a.append(i * m + j)
                nz_b.append(j * p + k)
                nz_c.append(k * n + i)
    nz_a = np.array(nz_a)
    nz_b = np.array(nz_b)
    nz_c = np.array(nz_c)
    
    def _objective_als(x, fixed_U=None, fixed_V=None, fixed_W=None):
        """
        Objective for ALS: fix two matrices, optimize the third.
        x is the flattened matrix being optimized.
        """
        if fixed_U is not None:
            U = fixed_U
            V = x[:b_size*rank].reshape(b_size, rank)
            W = x[b_size*rank:].reshape(c_size, rank)
        elif fixed_V is not None:
            U = x[:a_size*rank].reshape(a_size, rank)
            V = fixed_V
            W = x[a_size*rank:].reshape(c_size, rank)
        else:
            U = x[:a_size*rank].reshape(a_size, rank)
            V = x[a_size*rank:a_size*rank+b_size*rank].reshape(b_size, rank)
            W = x[a_size*rank+b_size*rank:].reshape(c_size, rank)
        
        # Compute residual only at non-zero entries
        # T[nz] = 1, recon[nz] = sum_r U[nz_a, r] * V[nz_b, r] * W[nz_c, r]
        recon_nz = np.einsum('ar,br,cr->r', U[nz_a, :], V[nz_b, :], W[nz_c, :])
        recon_nz = np.sum(recon_nz, axis=1)  # sum over r
        residual = 1.0 - recon_nz
        return np.sum(residual**2)
    
    start_time = time.time()
    max_restarts = 30
    max_als_iters = 50
    
    for attempt in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(seed + attempt * 1000)
        scale = 0.5 * (1 + attempt % 5)
        U = np.random.randn(a_size, rank) * scale
        V = np.random.randn(b_size, rank) * scale
        W = np.random.randn(c_size, rank) * scale
        
        # ALS iterations
        for als_iter in range(max_als_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Optimize U with V, W fixed
            x0 = U.flatten()
            try:
                res_U = minimize(lambda x: _objective_als(x, fixed_V=V, fixed_W=W), 
                                 x0, method='L-BFGS-B', 
                                 options={'maxiter': 30, 'ftol': 1e-10})
                U = res_U.x.reshape(a_size, rank)
            except Exception:
                pass
            
            # Optimize V with U, W fixed
            x0 = V.flatten()
            try:
                res_V = minimize(lambda x: _objective_als(x, fixed_U=U, fixed_W=W), 
                                 x0, method='L-BFGS-B', 
                                 options={'maxiter': 30, 'ftol': 1e-10})
                V = res_V.x.reshape(b_size, rank)
            except Exception:
                pass
            
            # Optimize W with U, V fixed
            x0 = W.flatten()
            try:
                res_W = minimize(lambda x: _objective_als(x, fixed_U=U, fixed_V=V), 
                                 x0, method='L-BFGS-B', 
                                 options={'maxiter': 30, 'ftol': 1e-10})
                W = res_W.x.reshape(c_size, rank)
            except Exception:
                pass
            
            # Check convergence
            if _verify_exact(T, U, V, W, tol=1e-6):
                return U, V, W
        
        # Final check after ALS
        if _verify_exact(T, U, V, W, tol=1e-6):
            return U, V, W
    
    return None

def _try_rank_gradient(T, n, m, p, rank, time_limit, seed=0):
    """
    Try to find a rank-`rank` decomposition using gradient-based optimization.
    Returns (U, V, W) if found, else None.
    """
    np.random.seed(seed)
    
    a_size = n * m
    b_size = m * p
    c_size = p * n
    
    # Precompute non-zero entries
    nz_a, nz_b, nz_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_a.append(i * m + j)
                nz_b.append(j * p + k)
                nz_c.append(k * n + i)
    nz_a = np.array(nz_a)
    nz_b = np.array(nz_b)
    nz_c = np.array(nz_c)
    
    def _flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def _unflatten(x):
        U = x[:a_size*rank].reshape(a_size, rank)
        V = x[a_size*rank:a_size*rank+b_size*rank].reshape(b_size, rank)
        W = x[a_size*rank+b_size*rank:].reshape(c_size, rank)
        return U, V, W
    
    def _objective(x):
        U, V, W = _unflatten(x)
        # Compute residual at non-zero entries
        recon_nz = np.einsum('ar,br,cr->r', U[nz_a, :], V[nz_b, :], W[nz_c, :])
        recon_nz = np.sum(recon_nz, axis=1)
        residual = 1.0 - recon_nz
        return np.sum(residual**2)
    
    start_time = time.time()
    max_restarts = 20
    
    for attempt in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(seed + attempt * 7)
        scale = 0.3 * (1 + attempt % 4)
        U = np.random.randn(a_size, rank) * scale
        V = np.random.randn(b_size, rank) * scale
        W = np.random.randn(c_size, rank) * scale
        
        x0 = _flatten(U, V, W)
        
        try:
            remaining = time_limit - (time.time() - start_time)
            maxiter = 100 if remaining > 5 else 30
            
            result = minimize(_objective, x0, method='L-BFGS-B', 
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 30})
            
            if result.fun < 1e-8:
                U, V, W = _unflatten(result.x)
                if _verify_exact(T, U, V, W, tol=1e-6):
                    return U, V, W
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    schoolbook_rank = n * m * p
    
    time_start = time.time()
    time_limit_total = 120  # Safety margin before 150s kill
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Lower bound: max(n*p, m*p, n*m) = max(16, 24, 6) = 24
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower bound up
    for rank in range(lower_bound, best_rank):
        elapsed = time.time() - time_start
        if elapsed > time_limit_total:
            break
        
        remaining_time = time_limit_total - elapsed
        # Give more time to lower ranks as they are harder
        per_try_time = min(remaining_time / max(1, best_rank - rank), 20.0)
        
        # Try ALS first (often more stable for tensor decomposition)
        result = _try_rank_als(T, n, m, p, rank, per_try_time * 0.6, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            break
        
        # Then try gradient-based
        result = _try_rank_gradient(T, n, m, p, rank, per_try_time * 0.4, seed=rank + 1000)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            break
    
    def _to_list(mat):
        res = []
        for row in mat:
            rl = []
            for val in row:
                ni = round(val)
                nh = round(2 * val) / 2.0
                if abs(val - ni) < 1e-6:
                    rl.append(int(ni))
                elif abs(val - nh) < 1e-6:
                    rl.append(nh)
                else:
                    rl.append(float(val))
            res.append(rl)
        return res
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    else:
        # Schoolbook fallback
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
