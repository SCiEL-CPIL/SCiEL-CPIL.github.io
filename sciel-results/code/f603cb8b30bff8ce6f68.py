import numpy as np
import time
import random
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _verify_exact(n, m, p, U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = U.shape[1]
    # Convert to Fractions
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_direct(n, m, p, target_rank, max_restarts=5, timeout=10):
    """
    Try to find a rank `target_rank` decomposition using direct optimization.
    """
    start_time = time.time()
    best_error = float('inf')
    best_x = None
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Indices for fast access
    # We need to compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def residuals(x):
        U = x[:num_u * target_rank].reshape(num_u, target_rank)
        V = x[num_u * target_rank : num_u * target_rank + num_v * target_rank].reshape(num_v, target_rank)
        W = x[num_u * target_rank + num_v * target_rank :].reshape(num_w, target_rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over c
                    # U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                    s = np.sum(prod)
                    res.append(s - 1.0)
        return res
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(42 + restart)
        # Initialize with small random values
        x0 = np.random.randn(num_u * target_rank + num_v * target_rank + num_w * target_rank) * 0.1
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if sol.cost < best_error:
                best_error = sol.cost
                best_x = sol.x
            if sol.cost < 1e-10:
                break
        except Exception:
            continue
            
    if best_x is None or best_error > 1e-8:
        return None
        
    U = best_x[:num_u * target_rank].reshape(num_u, target_rank)
    V = best_x[num_u * target_rank : num_u * target_rank + num_v * target_rank].reshape(num_v, target_rank)
    W = best_x[num_u * target_rank + num_v * target_rank :].reshape(num_w, target_rank)
    
    # Try to round to rationals
    for denom in [1, 2, 4, 8, 16]:
        U_r = np.round(U * denom) / denom
        V_r = np.round(V * denom) / denom
        W_r = np.round(W * denom) / denom
        
        # Convert to list of floats for verification
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        
        if _verify_exact(n, m, p, U_list, V_list, W_list):
            return U_list, V_list, W_list
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()

    # Try ranks from 49 down to 21
    for target_rank in range(49, 20, -1):
        if time.time() > deadline:
            break
            
        sol = _try_rank_direct(n, m, p, target_rank, max_restarts=5, timeout=10)
        
        if sol is not None:
            best_U, best_V, best_W = sol
            best_rank = target_rank
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
