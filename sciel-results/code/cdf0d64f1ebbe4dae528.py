import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # 40s margin before 150s limit
    
    # Target tensor
    T = np.ones((n, m, p))
    
    # Schoolbook decomposition (rank 63)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for c in range(r):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            T_recon += u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_mat[np.newaxis, np.newaxis, :]
        return np.allclose(T_recon, T, atol=tol)
    
    def snap_to_half_int(arr):
        return np.round(arr * 2) / 2
    
    # Verify schoolbook works
    if not verify(U_school, V_school, W_school):
        raise Exception("Schoolbook decomposition failed verification")
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try ranks from 62 down to 50
    # For each rank, try a few random initializations with Nelder-Mead
    for r in range(rank_school - 1, max(rank_school - 15, 45), -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try 3 random initializations per rank
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + r + attempt)  # Reproducible
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            def objective(params):
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                u_reshaped = U_flat.reshape(n, m, r)
                v_reshaped = V_flat.reshape(m, p, r)
                w_reshaped = W_flat.reshape(p, n, r)
                
                T_recon = np.einsum('ijc,jkc,kic->ijk', u_reshaped, v_reshaped, w_reshaped)
                
                diff = T_recon - T
                return np.sum(diff**2)
            
            params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, params_init, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                U_snapped = snap_to_half_int(U_opt)
                V_snapped = snap_to_half_int(V_opt)
                W_snapped = snap_to_half_int(W_opt)
                
                if verify(U_snapped, V_snapped, W_snapped):
                    best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
