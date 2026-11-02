import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                T[i, :, k] = 1.0

    def reconstruct(U, V, W):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        return np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)

    def loss(params, R):
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        T_rec = reconstruct(U, V, W)
        err = np.sum((T - T_rec)**2)
        return err

    def try_rank(R, max_iters=10):
        for i in range(max_iters):
            if time.time() > deadline:
                return None
            
            init_params = np.random.uniform(-1, 1, size=(n*m + m*p + p*n) * R)
            
            try:
                res = minimize(loss, init_params, args=(R,), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5})
            except Exception:
                continue
            
            if res.fun < 1e-4:
                U_flat = res.x[:n*m*R]
                V_flat = res.x[n*m*R:n*m*R + m*p*R]
                W_flat = res.x[n*m*R + m*p*R:]
                
                U_rounded = np.round(2 * U_flat) / 2.0
                V_rounded = np.round(2 * V_flat) / 2.0
                W_rounded = np.round(2 * W_flat) / 2.0
                
                U_r = U_rounded.reshape(n*m, R)
                V_r = V_rounded.reshape(m*p, R)
                W_r = W_rounded.reshape(p*n, R)
                
                T_rec_r = reconstruct(U_r, V_r, W_r)
                err_r = np.sum((T - T_rec_r)**2)
                
                if err_r < 1e-6:
                    return U_r, V_r, W_r, R
        
        return None

    # Try ranks from 104 down to 95
    for R in range(104, 94, -1):
        if time.time() > deadline:
            break
        res = try_rank(R)
        if res:
            U_r, V_r, W_r, R_found = res
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
