"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook-1 down to some minimum
    for rank in range(schoolbook_rank - 1, 14, -1):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                continue
        except Exception:
            continue
    
    return best_result

def _verify_solution(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    return max_err < tol

def _snap_to_half_integers(U, V, W, n, m, p):
    """Try snapping to half-integers and verify."""
    for denom in [1, 2, 3, 4, 6, 8, 12]:
        U_snap = np.round(U * denom) / denom
        V_snap = np.round(V * denom) / denom
        W_snap = np.round(W * denom) / denom
        
        if _verify_solution(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
    return None, None, None

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank using least_squares."""
    num_params = (n * m + m * p + p * n) * rank
    um = n * m
    vm = m * p
    wm = p * n
    
    # Residual function for least_squares
    def residual_function(params):
        U = params[:um*rank].reshape(um, rank)
        V = params[um*rank:um*rank + vm*rank].reshape(vm, rank)
        W = params[um*rank + vm*rank:].reshape(wm, rank)
        
        residuals = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals

    # Multiple restarts to escape local minima
    np.random.seed(int(time.time() * 1000) % 10000)
    
    num_restarts = 20 if rank >= 20 else 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        U0 = np.random.randn(um, rank)
        V0 = np.random.randn(vm, rank)
        W0 = np.random.randn(wm, rank)
        
        # Scale initialization
        scale = 1.0 / np.sqrt(rank)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residual_function, params0, method='trf', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        except Exception:
            continue
            
        if result.cost < 1e-6:
            U_opt = result.x[:um*rank].reshape(um, rank)
            V_opt = result.x[um*rank:um*rank + vm*rank].reshape(vm, rank)
            W_opt = result.x[um*rank + vm*rank:].reshape(wm, rank)
            
            # Try snapping to half-integers
            U_snap, V_snap, W_snap = _snap_to_half_integers(U_opt, V_opt, W_opt, n, m, p)
            if U_snap is not None:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
    
    return None

# EVOLVE-BLOCK-END
