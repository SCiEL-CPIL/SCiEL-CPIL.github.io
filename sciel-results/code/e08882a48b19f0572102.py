import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Check: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    error = np.linalg.norm(T - 1.0)
    return error

def _snap_to_half(arr):
    return np.round(arr * 2) / 2

def _local_search_snapping(U, V, W, n, m, p, max_iters=10):
    """
    Attempt to snap to half-integers and refine via local search.
    """
    U_s = _snap_to_half(U)
    V_s = _snap_to_half(V)
    W_s = _snap_to_half(W)
    
    best_err = _verify(U_s, V_s, W_s, n, m, p)
    
    if best_err > 1e-3:
        return U_s, V_s, W_s, best_err

    for _ in range(max_iters):
        improved = False
        
        def try_update(mat):
            nonlocal improved, best_err
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    orig_val = mat[i, j]
                    # Try neighbors in the half-integer grid
                    candidates = [orig_val - 0.5, orig_val + 0.5]
                    for cand in candidates:
                        mat[i, j] = cand
                        err = _verify(U_s, V_s, W_s, n, m, p)
                        if err < best_err - 1e-9:
                            best_err = err
                            improved = True
                        else:
                            mat[i, j] = orig_val
                            
        try_update(U_s)
        try_update(V_s)
        try_update(W_s)
        
        if not improved:
            break
            
    return U_s, V_s, W_s, best_err

def _try_rank(R, n, m, p, time_limit=10, num_restarts=5):
    best_error = float('inf')
    best_factors = None
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
            U_mat = U_flat.reshape(n, m, R)
            V_mat = V_flat.reshape(m, p, R)
            W_mat = W_flat.reshape(p, n, R)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
            err = np.sum((T - 1.0)**2)
            return err
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            x_opt = res.x
            U_flat, V_flat, W_flat = x_opt[:n*m*R], x_opt[n*m*R:n*m*R+m*p*R], x_opt[n*m*R+m*p*R:]
            U_opt = U_flat.reshape(n*m, R)
            V_opt = V_flat.reshape(m*p, R)
            W_opt = W_flat.reshape(p*n, R)
            
            # Snap and verify
            U_snap = _snap_to_half(U_opt)
            V_snap = _snap_to_half(V_opt)
            W_snap = _snap_to_half(W_opt)
            
            err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
            
            if err_snap < 1e-6:
                return U_snap, V_snap, W_snap
                
            if err_snap < best_error:
                best_error = err_snap
                best_factors = (U_snap, V_snap, W_snap)
                
        except Exception:
            pass
            
    return None

def _try_subtensor_rank_11():
    """
    Try to find rank 11 decomposition for M_{2,2,3}.
    Returns (U, V, W) for the sub-tensor or None.
    n=2, m=2, p=3.
    """
    n, m, p = 2, 2, 3
    R = 11
    # Use a slightly longer time limit as this is a fallback
    res = _try_rank(R, n, m, p, time_limit=15, num_restarts=10)
    return res

def construct():
    n, m, p = DIMENSIONS
    global_start = time.time()
    deadline = global_start + 130 # Safety margin
    
    # Try ranks in ascending order of difficulty (lower rank first)
    # Allocate time budget based on remaining time
    ranks_to_try = [12, 13, 14, 15, 16, 17]
    
    # Dynamic time allocation
    # Prioritize lower ranks with more restarts if time permits
    time_budget = deadline - time.time() - 5 # 5s safety margin
    
    for R in ranks_to_try:
        if time.time() >= deadline - 5:
            break
            
        # Estimate time per rank: lower ranks are harder, give them more time
        # Simple heuristic: 1/3 of remaining for first 2, 1/5 for next 2, rest for others
        if R <= 13:
            t_limit = min(15, time_budget * 0.3)
            n_restarts = 10
        elif R <= 15:
            t_limit = min(10, time_budget * 0.2)
            n_restarts = 5
        else:
            t_limit = min(5, time_budget * 0.1)
            n_restarts = 3
            
        if t_limit < 1:
            continue
            
        res = _try_rank(R, n, m, p, time_limit=t_limit, num_restarts=n_restarts)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
