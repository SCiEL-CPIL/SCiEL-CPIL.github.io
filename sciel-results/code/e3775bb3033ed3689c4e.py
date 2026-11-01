"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _create_objective(n, m, p, T_exp):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    def objective(params):
        rank = params.shape[0] // (dim_a + dim_b + dim_c)
        if rank == 0: return 1e10
        
        U_flat = params[:rank*dim_a]
        V_flat = params[rank*dim_a:2*rank*dim_a]
        W_flat = params[2*rank*dim_a:3*rank*dim_a]
        
        U = U_flat.reshape(rank, dim_a).T
        V = V_flat.reshape(rank, dim_b).T
        W = W_flat.reshape(rank, dim_c).T
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - T_exp
        return np.sum(diff**2)
    return objective

def _try_rank_scipy(target_rank, n, m, p, deadline, max_attempts=100, seed=42):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    total_params = target_rank * (dim_a + dim_b + dim_c)
    
    T_exp = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
                
    objective = _create_objective(n, m, p, T_exp)
    
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Random initialization
        scale = 1.0 / np.sqrt(target_rank)
        params = np.random.randn(total_params) * scale
        
        # Run optimization
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                              options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
        except Exception:
            continue
        
        if result.fun < 1e-5: # Check if solution is good enough
            # Extract U, V, W
            rank = target_rank
            U = result.x[:rank*dim_a].reshape(rank, dim_a).T
            V = result.x[rank*dim_a:2*rank*dim_a].reshape(rank, dim_b).T
            W = result.x[2*rank*dim_a:3*rank*dim_a].reshape(rank, dim_c).T
            
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Try ranks in increasing order of difficulty (lowest rank first)
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank_scipy(target_rank, n, m, p, deadline, max_attempts=100, seed=42+target_rank)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
