import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Try rank 42 (3 * 14)
    found = _try_rank(n, m, p, 42, deadline)
    if found:
        return found
        
    # Try rank 36
    found = _try_rank(n, m, p, 36, deadline)
    if found:
        return found
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _try_rank(n, m, p, rank, deadline):
    # Use least_squares to find U, V, W
    # Variables: U (nm x rank), V (mp x rank), W (pn x rank)
    # Total variables: rank * (nm + mp + pn)
    
    def residual_func(params):
        # Reshape params to U, V, W
        nm, mp, pn = n*m, m*p, p*n
        U = params[:rank*nm].reshape(nm, rank)
        V = params[rank*nm:rank*nm+rank*mp].reshape(mp, rank)
        W = params[rank*nm+rank*mp:].reshape(pn, rank)
        
        # Compute residual
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1.0
        return res.flatten()
    
    # Initialize with schoolbook structure if possible, else random
    # For rank 42, we can initialize with 3 blocks of 14
    # But we don't have rank 14 coeffs.
    # We initialize with random values.
    
    nm, mp, pn = n*m, m*p, p*n
    total_vars = rank * (nm + mp + pn)
    
    best_res = float('inf')
    best_params = None
    
    for trial in range(10):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(42 + trial)
        init_params = np.random.randn(total_vars) * 0.1
        
        try:
            result = least_squares(residual_func, init_params, method='lm', max_nfev=10000)
            res_norm = np.max(np.abs(result.fun))
            if res_norm < 1e-9:
                # Check if we can round to half-integers
                U, V, W = _extract_UVW(result.x, n, m, p, rank)
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                res_r = _compute_residual_fast(U_r, V_r, W_r, n, m, p)
                if np.max(np.abs(res_r)) < 1e-6:
                    return _finalize(U_r, V_r, W_r, n, m, p)
                else:
                    return _finalize(U, V, W, n, m, p)
            
            if res_norm < best_res:
                best_res = res_norm
                best_params = result.x
        except Exception as e:
            pass
            
    return None

def _extract_UVW(params, n, m, p, rank):
    nm, mp, pn = n*m, m*p, p*n
    U = params[:rank*nm].reshape(nm, rank)
    V = params[rank*nm:rank*nm+rank*mp].reshape(mp, rank)
    W = params[rank*nm+rank*mp:].reshape(pn, rank)
    return U, V, W

def _compute_residual_fast(U, V, W, n, m, p):
    res = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1.0
    return res

def _finalize(U, V, W, n, m, p):
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
