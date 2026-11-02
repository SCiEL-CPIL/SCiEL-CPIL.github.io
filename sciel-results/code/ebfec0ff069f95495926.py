import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank): return False
    if V_arr.shape != (m*p, rank): return False
    if W_arr.shape != (p*n, rank): return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j, :]
                v_row = V_arr[j * p + k, :]
                w_row = W_arr[k * n + i, :]
                product = np.sum(u_row * v_row * w_row)
                if abs(product - 1.0) > 1e-9:
                    return False
    return True

def _optimize_rank(n, m, p, target_rank, time_limit_per_rank=10.0):
    start_time = time.time()
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    # Pre-compute index arrays for vectorization
    idx_u = np.array([x[0] for x in indices])
    idx_v = np.array([x[1] for x in indices])
    idx_w = np.array([x[2] for x in indices])
    
    def objective(params):
        size_u = U_dim * target_rank
        size_v = V_dim * target_rank
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:]
        
        U = U_flat.reshape(U_dim, target_rank)
        V = V_flat.reshape(V_dim, target_rank)
        W = W_flat.reshape(W_dim, target_rank)
        
        # Vectorized computation
        # U[idx_u, :] has shape (N, R)
        # V[idx_v, :] has shape (N, R)
        # W[idx_w, :] has shape (N, R)
        # Product is element-wise, then sum over R
        vals = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :], axis=1)
        errors = vals - 1.0
        return np.sum(errors * errors)

    best_result = None
    best_loss = float('inf')
    
    num_restarts = 50 # Increase restarts due to speedup
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit_per_rank:
            break
            
        # Sparse random initialization
        # Probability of non-zero
        p_nonzero = 0.3
        U0 = np.random.randn(U_dim, target_rank) * (np.random.rand(U_dim, target_rank) < p_nonzero)
        V0 = np.random.randn(V_dim, target_rank) * (np.random.rand(V_dim, target_rank) < p_nonzero)
        W0 = np.random.randn(W_dim, target_rank) * (np.random.rand(W_dim, target_rank) < p_nonzero)
        
        # Ensure some non-zeros
        if not np.any(U0): U0[0,0] = 1
        if not np.any(V0): V0[0,0] = 1
        if not np.any(W0): W0[0,0] = 1
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.fun < best_loss:
                best_loss = res.fun
                if best_loss < 1e-6:
                    size_u = U_dim * target_rank
                    size_v = V_dim * target_rank
                    U_sol = res.x[:size_u].reshape(U_dim, target_rank)
                    V_sol = res.x[size_u:size_u + size_v].reshape(V_dim, target_rank)
                    W_sol = res.x[size_u + size_v:].reshape(W_dim, target_rank)
                    
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                        
        except Exception:
            continue
            
    return None

def _find_best_decomposition(n, m, p, time_limit=110.0):
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    def get_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return U, V, W

    best_U, best_V, best_W = get_schoolbook()
    best_rank = schoolbook_rank

    # Try ranks 27, 28, 29
    # 27 is likely minimal based on bounds
    ranks_to_try = [27, 28, 29]
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > time_limit - 10:
            break
            
        time_slice = 30.0
        result = _optimize_rank(n, m, p, target_rank, time_limit_per_rank=time_slice)
        
        if result is not None:
            U_list, V_list, W_list = result
            if _verify_decomposition(U_list, V_list, W_list):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break
    
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    random.seed(42)
    np.random.seed(42)
    
    try:
        U, V, W = _find_best_decomposition(n, m, p, time_limit=110.0)
    except Exception:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
