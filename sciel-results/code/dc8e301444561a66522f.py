import numpy as np
import scipy.optimize
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for row in range(n):
        for mid in range(m):
            for col in range(p):
                i = row * m + mid
                j = mid * p + col
                l = col * n + row
                T[i, j, l] = 1.0
                
    T_flat = T.flatten()
    
    def compute_residual(U, V, W):
        res = np.zeros_like(T)
        for k in range(U.shape[1]):
            res += np.einsum('i,j,l->ijl', U[:, k], V[:, k], W[:, k])
        return (T - res).flatten()
        
    def snap_to_half_int(X):
        return np.round(X * 2) / 2.0
        
    def is_valid(U, V, W):
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        res = np.zeros_like(T)
        for k in range(U.shape[1]):
            res += np.einsum('i,j,l->ijl', U[:, k], V[:, k], W[:, k])
        return np.allclose(T - res, 0, atol=1e-9)
        
    start_time = time.time()
    time_limit = 145.0
    
    # Try ranks from low to high
    for r in range(28, 54):
        rank_start_time = time.time()
        max_time_per_rank = 2.0
        
        # Number of restarts based on time available
        # Estimate 0.05s per restart
        max_restarts = int(max_time_per_rank / 0.05)
        if max_restarts < 1: max_restarts = 1
        
        for restart in range(max_restarts):
            if time.time() - rank_start_time > max_time_per_rank:
                break
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(dim_u, r) * 0.1
            V = np.random.randn(dim_v, r) * 0.1
            W = np.random.randn(dim_w, r) * 0.1
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_x = x[:dim_u*r].reshape(dim_u, r)
                V_x = x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                W_x = x[dim_u*r+dim_v*r:].reshape(dim_w, r)
                return compute_residual(U_x, V_x, W_x)
                
            try:
                res_opt = scipy.optimize.least_squares(objective, x0, method='lm', max_nfev=200)
                x_sol = res_opt.x
                U_sol = x_sol[:dim_u*r].reshape(dim_u, r)
                V_sol = x_sol[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                W_sol = x_sol[dim_u*r+dim_v*r:].reshape(dim_w, r)
                
                if res_opt.cost < 1e-10:
                    U_s = snap_to_half_int(U_sol)
                    V_s = snap_to_half_int(V_sol)
                    W_s = snap_to_half_int(W_sol)
                    if is_valid(U_s, V_s, W_s):
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            except Exception:
                pass
                
    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
