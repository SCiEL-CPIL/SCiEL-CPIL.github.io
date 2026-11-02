import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    # Precompute schoolbook basis vectors for U, V, W
    num_basis = n * m * p
    U_basis = np.zeros((num_basis, n * m))
    V_basis = np.zeros((num_basis, m * p))
    W_basis = np.zeros((num_basis, p * n))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_basis[idx, i * m + j] = 1.0
                V_basis[idx, j * p + k] = 1.0
                W_basis[idx, k * n + i] = 1.0
                idx += 1

    def residual_fn(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, R)).T # (R, n*m)
        V = V_flat.reshape((m * p, R)).T # (R, m*p)
        W = W_flat.reshape((p * n, R)).T # (R, p*n)
        
        U_reshaped = U.reshape((R, n, m))
        V_reshaped = V.reshape((R, m, p))
        W_reshaped = W.reshape((R, p, n))
        
        recon = np.zeros_like(T)
        for r in range(R):
            u = U_reshaped[r]
            v = V_reshaped[r]
            w = W_reshaped[r]
            
            uv = np.einsum('ij,jk->ijjk', u, v)
            recon += np.einsum('ijjk,ab->ijkab', uv, w)
            
        return (recon.flatten() - T_flat)

    def schoolbook_solution():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from 45 to 83
    # Lower bound is roughly max(n*m, m*p, n*p) = 28
    for R in range(45, min(84, n*m*p)):
        if time.time() > deadline:
            break
            
        num_restarts = 10 if R < 60 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            if restart == 0:
                # Initialization: Random linear combination of schoolbook basis
                C_U = np.random.randn(num_basis, R) * 0.5
                C_V = np.random.randn(num_basis, R) * 0.5
                C_W = np.random.randn(num_basis, R) * 0.5
                
                U_init = U_basis.T @ C_U
                V_init = V_basis.T @ C_V
                W_init = W_basis.T @ C_W
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                x0 = np.random.randn(total_size) * 0.5
            
            try:
                res = least_squares(residual_fn, x0, args=(R,), method='lm', max_nfev=500)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                x_snapped = np.concatenate([U_snap, V_snap, W_snap])
                res_snapped = residual_fn(x_snapped, R)
                
                if np.max(np.abs(res_snapped)) < 1e-9:
                    best_rank = R
                    best_solution = {
                        "U": U_snap.reshape((n*m, R)).tolist(),
                        "V": V_snap.reshape((m*p, R)).tolist(),
                        "W": W_snap.reshape((p*n, R)).tolist()
                    }
                    break
        if best_solution:
            break

    if best_solution:
        return best_solution
    else:
        return schoolbook_solution()
