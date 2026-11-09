import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is exact. U, V, W are numpy arrays."""
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    except Exception:
        return False
    return np.allclose(T, 1.0, atol=tol, rtol=0.0)

def _try_find_lower_rank(n, m, p, target_rank, time_limit, seed=42):
    """
    Attempts to find a decomposition of rank target_rank.
    Returns (U, V, W) if found and verified, else None.
    """
    np.random.seed(seed)
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank

    def cost_and_grad(params):
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n, m, target_rank)
        V = V_flat.reshape(m, p, target_rank)
        W = W_flat.reshape(p, n, target_rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W, optimize=True)
        err = T - 1.0
        cost = np.sum(err**2)
        
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', err, V, W, optimize=True)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', err, U, W, optimize=True)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', err, U, V, optimize=True)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return cost, grad

    best_result = None
    best_cost = float('inf')
    start_time = time.time()
    
    for restart in range(3):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randn(n, m, target_rank) * 0.5
        V = np.random.randn(m, p, target_rank) * 0.5
        W = np.random.randn(p, n, target_rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(cost_and_grad, params, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'maxfun': 200})
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res.x
                
                if best_cost < 1e-4:
                    U_flat = best_result[:size_U]
                    V_flat = best_result[size_U:size_U + size_V]
                    W_flat = best_result[size_U + size_V:]
                    
                    U_sol = U_flat.reshape(n, m, target_rank).reshape(n*m, target_rank)
                    V_sol = V_flat.reshape(m, p, target_rank).reshape(m*p, target_rank)
                    W_sol = W_flat.reshape(p, n, target_rank).reshape(p*n, target_rank)
                    
                    if _verify(U_sol, V_sol, W_sol, n, m, p):
                        return U_sol, V_sol, W_sol
                        
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    for target_rank in range(best_rank - 1, best_rank - 6, -1):
        if time.time() > deadline:
            break
            
        time_limit = min(10.0, deadline - time.time())
        if time_limit < 1.0:
            break
            
        result = _try_find_lower_rank(n, m, p, target_rank, time_limit)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
