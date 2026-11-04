import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid product index."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _residual(params, n, m, p, rank, T_flat):
    """Compute the residual vector for least squares."""
    U = params[:n * m * rank].reshape(n * m, rank)
    V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
    W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
    
    # Compute reconstruction: sum_c U[:,c] * V[:,c] * W[:,c]
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return (recon - T_flat.reshape(n * m, m * p, p * n)).flatten()

def _snap_to_half_integers(params, n, m, p, rank, T_flat, tol=1e-5):
    """Snap parameters to multiples of 0.5 and verify."""
    U = params[:n * m * rank].reshape(n * m, rank)
    V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
    W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
    
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    err = np.sum((recon - T_flat.reshape(n * m, m * p, p * n)) ** 2)
    
    if err < tol:
        return U_snap, V_snap, W_snap
    return None

def _try_rank(n, m, p, rank, T_flat, time_budget, max_restarts=100):
    """Try to find a rank-rank decomposition using least_squares."""
    start_time = time.time()
    n_params = (n * m + m * p + p * n) * rank
    
    best_err = np.inf
    best_params = None
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget * 0.9:
            break
        
        # Random initialization
        # Use different scales to help escape local minima
        scale = np.random.uniform(0.1, 2.0)
        params_init = np.random.randn(n_params) * scale
        
        try:
            result = least_squares(
                _residual,
                params_init,
                args=(n, m, p, rank, T_flat),
                method='trf',
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15,
                max_nfev=1000
            )
            
            err = np.sum(result.fun**2)
            if err < best_err:
                best_err = err
                best_params = result.x.copy()
                
                if err < 1e-8:
                    break
                    
        except Exception:
            continue
            
    return best_err, best_params

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Deadline guard
    total_start = time.time()
    time_budget = 110.0
    
    # Fallback: Schoolbook rank 24
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try ranks from 18 up to 23
    # Known lower bound for 2x3x4 is 18
    for rank in range(18, rank_fb):
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        
        # Allocate more time to lower ranks as they are more likely optimal and harder
        # But we want to find 18 first.
        rank_budget = remaining * 0.4 if rank == 18 else remaining * 0.2
        if rank_budget < 1.0:
            rank_budget = 1.0
            
        best_err, best_params = _try_rank(n, m, p, rank, T_flat, rank_budget)
        
        if best_err < 1e-6 and best_params is not None:
            snapped = _snap_to_half_integers(best_params, n, m, p, rank, T_flat)
            if snapped is not None:
                U_snap, V_snap, W_snap = snapped
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            # If snapping fails but error is very small, return float version
            # The evaluator might accept close floats? "integer/half-integer entries" is specified.
            # So we MUST snap successfully.
            
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
