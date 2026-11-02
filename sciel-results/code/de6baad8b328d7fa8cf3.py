"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W form a valid decomposition."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], np.dot(V[j*p+k], W[k*n+i]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomp(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U.tolist(), V.tolist(), W.tolist()

def _objective(params, n, m, p, rank):
    """Objective: sum of (actual - 1)^2 over all (i,j,k)."""
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = params[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    obj = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], np.dot(V[j*p+k], W[k*n+i]))
                obj += (s - 1.0)**2
    return obj

def _try_rank_lbfgs(rank, n, m, p, max_iters=2000, seed=42, time_limit=8):
    """Try to find a rank-rank decomposition using L-BFGS-B optimization."""
    rng = np.random.RandomState(seed)
    
    U0 = rng.randn(n * m, rank) * 0.5
    V0 = rng.randn(m * p, rank) * 0.5
    W0 = rng.randn(p * n, rank) * 0.5
    
    params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def obj_func(params):
        if time.time() > _START_TIME + time_limit:
            return 1e10
        return _objective(params, n, m, p, rank)
    
    result = minimize(obj_func, params0, method='L-BFGS-B', 
                      options={'maxiter': max_iters, 'maxfun': max_iters*2,
                               'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    if result.fun < 1e-6:
        U = result.x[:n*m*rank].reshape(n*m, rank)
        V = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = result.x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
        if _verify(U, V, W, n, m, p, tol=1e-5):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    lower_bound = max(n*m, m*p, n*p)
    
    target_ranks = [schoolbook_rank - 1, schoolbook_rank - 2, schoolbook_rank - 3]
    
    for rank in target_ranks:
        if rank < lower_bound:
            continue
        if time.time() > _DEADLINE - 15:
            break
            
        for seed in range(8):
            if time.time() > _DEADLINE - 15:
                break
            result = _try_rank_lbfgs(rank, n, m, p, max_iters=1000, seed=seed, time_limit=8)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
                    
    if best_U is None:
        U, V, W = _schoolbook_decomp(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
