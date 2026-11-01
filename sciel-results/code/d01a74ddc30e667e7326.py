"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T.reshape(-1)
    
    def _objective(params, rank):
        sz_u = n*m*rank
        sz_v = m*p*rank
        sz_w = p*n*rank
        U = params[:sz_u].reshape(n*m, rank)
        V = params[sz_u:sz_u+sz_v].reshape(m*p, rank)
        W = params[sz_u+sz_v:sz_u+sz_v+sz_w].reshape(p*n, rank)
        
        # Compute T_approx
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx.reshape(-1) - T_flat
        return np.sum(diff*diff)
    
    def _objective_grad(params, rank):
        sz_u = n*m*rank
        sz_v = m*p*rank
        sz_w = p*n*rank
        U = params[:sz_u].reshape(n*m, rank)
        V = params[sz_u:sz_u+sz_v].reshape(m*p, rank)
        W = params[sz_u+sz_v:sz_u+sz_v+sz_w].reshape(p*n, rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T
        
        # Gradients
        gU = np.einsum('abc,br,cr->ar', diff, V, W)
        gV = np.einsum('abc,ar,cr->br', diff, U, W)
        gW = np.einsum('abc,ar,br->cr', diff, U, V)
        
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
    
    def _verify(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < 1e-9

    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p # 18
    
    # Try ranks from 13 up to 17
    # Rank 13 is the theoretical minimum for 2x3x3
    rng = np.random.default_rng(42)
    
    for rank in range(13, 18):
        if _remaining() < 10:
            break
            
        found = False
        # Number of restarts depends on remaining time
        # Estimate time per restart
        # For rank 13, params = 6*13 + 9*13 + 6*13 = 78*13 = 1014 params
        # L-BFGS-B is fast for this size.
        max_restarts = 100
        if rank > 13:
            max_restarts = 200
            
        for restart in range(max_restarts):
            if _remaining() < 5:
                break
                
            # Random initialization
            U = rng.standard_normal((n*m, rank)) * 0.5
            V = rng.standard_normal((m*p, rank)) * 0.5
            W = rng.standard_normal((p*n, rank)) * 0.5
            params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=lambda p, r: _objective_grad(p, r),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-14}
                )
                
                if _remaining() < 5:
                    break
                    
                final_obj = res.fun
                if final_obj < 1e-10:
                    U, V, W = res.x[:n*m*rank].reshape(n*m, rank), \
                               res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank), \
                               res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Check if we can round to half-integers
                    # The problem requires integer/half-integer entries
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if _verify(U_rounded, V_rounded, W_rounded):
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        best_rank = rank
                        found = True
                        break
                    else:
                        # If rounding fails, maybe the solution is not exactly on the lattice?
                        # But for matrix multiplication, it usually is.
                        # If it's not, we might need a higher rank or different initialization.
                        # For now, we reject it.
                        pass
                        
            except Exception:
                pass
        
        if found:
            # Found a valid decomposition at this rank; we want the minimal rank
            # So we break immediately if we found rank 13.
            # If we found rank 14, we should have tried 13 first.
            # Since we iterate 13, 14, ..., we break on first success.
            break
    
    # If we didn't find a better rank, return schoolbook
    if best_rank == n * m * p:
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists of floats (which will be integers or half-integers)
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
