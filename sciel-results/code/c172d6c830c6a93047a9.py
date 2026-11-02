import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    # Precompute schoolbook basis vectors
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
        
        U = U_flat.reshape((n * m, R)).T 
        V = V_flat.reshape((m * p, R)).T
        W = W_flat.reshape((p * n, R)).T
        
        recon = np.einsum('rij,rab,rcd->ijabc', U, V, W)
        recon_matched = recon.transpose(0, 1, 2, 3, 5, 4)
        
        return (recon_matched.flatten() - T_flat)
    
    def check_and_snap(res_x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = res_x[:size_U]
        V_flat = res_x[size_U:size_U + size_V]
        W_flat = res_x[size_U + size_V:]
        
        # Try integers
        U_snap = np.round(U_flat)
        V_snap = np.round(V_flat)
        W_snap = np.round(W_flat)
        
        x_snapped = np.concatenate([U_snap, V_snap, W_snap])
        res_snapped = residual_fn(x_snapped, R)
        
        if np.max(np.abs(res_snapped)) < 1e-9:
            return True, U_snap, V_snap, W_snap
        
        # Try half-integers
        U_snap = np.round(U_flat * 2) / 2
        V_snap = np.round(V_flat * 2) / 2
        W_snap = np.round(W_flat * 2) / 2
        
        x_snapped = np.concatenate([U_snap, V_snap, W_snap])
        res_snapped = residual_fn(x_snapped, R)
        
        if np.max(np.abs(res_snapped)) < 1e-9:
            return True, U_snap, V_snap, W_snap
        
        return False, None, None, None

    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from 45 to 83
    for R in range(45, min(84, n*m*p)):
        if time.time() > deadline:
            break
            
        num_restarts = 8 if R < 55 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            if restart == 0:
                # Structured initialization
                C_U = np.random.randn(num_basis, R) * 0.3
                C_V = np.random.randn(num_basis, R) * 0.3
                C_W = np.random.randn(num_basis, R) * 0.3
                
                U_init = U_basis.T @ C_U
                V_init = V_basis.T @ C_V
                W_init = W_basis.T @ C_W
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            elif restart == 1:
                # Scaled random
                x0 = np.random.randn(total_size) * 0.3
            else:
                # Sparse random
                x0 = np.random.randn(total_size)
                mask = np.random.random(total_size) < 0.3
                x0[~mask] = 0
                x0[mask] *= 0.5
            
            try:
                res = least_squares(residual_fn, x0, args=(R,), method='lm', max_nfev=800, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                found, U_s, V_s, W_s = check_and_snap(res.x, R)
                
                if found:
                    best_rank = R
                    best_solution = {
                        "U": U_s.reshape((n*m, R)).tolist(),
                        "V": V_s.reshape((m*p, R)).tolist(),
                        "W": W_s.reshape((p*n, R)).tolist()
                    }
                    break
        if best_solution:
            break

    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook
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
