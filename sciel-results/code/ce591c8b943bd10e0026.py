import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    T_flat = T.flatten()
    
    def residuals(params, rank):
        size_u = dim_u * rank
        size_v = dim_v * rank
        size_w = dim_w * rank
        U = params[:size_u].reshape(dim_u, rank)
        V = params[size_u:size_u+size_v].reshape(dim_v, rank)
        W = params[size_u+size_v:size_u+size_v+size_w].reshape(dim_w, rank)
        R = np.einsum('uc,vc,wc->uvw', U, V, W)
        return (R.flatten() - T_flat)
        
    def try_rank(rank, max_restarts=20, timeout=10):
        start_time = time.time()
        size_u = dim_u * rank
        size_v = dim_v * rank
        size_w = dim_w * rank
        total_size = size_u + size_v + size_w
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            params = np.random.randint(-2, 3, size=total_size).astype(float)
            try:
                res = least_squares(residuals, params, args=(rank,), method='trf', max_nfev=500)
            except Exception:
                continue
            if res.cost < 1e-6:
                U = res.x[:size_u].reshape(dim_u, rank)
                V = res.x[size_u:size_u+size_v].reshape(dim_v, rank)
                W = res.x[size_u+size_v:size_u+size_v+size_w].reshape(dim_w, rank)
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                R_check = np.einsum('uc,vc,wc->uvw', U_rounded, V_rounded, W_rounded)
                if np.allclose(R_check, T, atol=1e-9):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        return None

    deadline = time.time() + 110
    for rank in range(19, 25):
        if time.time() > deadline:
            break
        result = try_rank(rank, timeout=min(10, deadline - time.time()))
        if result:
            return result
            
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
