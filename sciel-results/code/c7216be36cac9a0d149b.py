import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # 1. Generate the Schoolbook Decomposition (rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = get_schoolbook()
    
    # Verify schoolbook decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_sb[i*m+j, :] * V_sb[j*p+k, :] * W_sb[k*n+i, :])
                if not np.isclose(s, 1.0):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    best_result = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    best_rank = n * m * p
    
    # Try to find lower rank decompositions via nonlinear optimization
    def try_rank(r, max_iters=300):
        if time.time() - start_time > deadline - 10:
            return None
        
        # Initialize with schoolbook truncated
        U_init = U_sb[:, :r].copy()
        V_init = V_sb[:, :r].copy()
        W_init = W_sb[:, :r].copy()
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(params):
            if time.time() - start_time > deadline - 5:
                return None
            U_p = params[:n*m*r].reshape(n*m, r)
            V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_p[i*m+j, :] * V_p[j*p+k, :] * W_p[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        # Try nonlinear least squares with multiple random restarts
        for restart in range(5):
            if time.time() - start_time > deadline - 15:
                return None
            
            if restart == 0:
                params0 = params_init
            else:
                np.random.seed(42 + restart)
                U_p = U_init + 0.5 * np.random.randn(n*m, r)
                V_p = V_init + 0.5 * np.random.randn(m*p, r)
                W_p = W_init + 0.5 * np.random.randn(p*n, r)
                params0 = np.concatenate([U_p.flatten(), V_p.flatten(), W_p.flatten()])
            
            try:
                result = least_squares(residuals, params0, method='trf', 
                                      max_nfev=max_iters, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
            
            if result.cost < 1e-10:
                U_p = result.x[:n*m*r].reshape(n*m, r)
                V_p = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_p = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Round to nearest half-integer
                U_r = np.round(U_p * 2) / 2
                V_r = np.round(V_p * 2) / 2
                W_r = np.round(W_p * 2) / 2
                
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                            if not np.isclose(s, 1.0, atol=1e-9):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return (U_r, V_r, W_r)
        
        return None
    
    # Try ranks from 44 down to 20
    for r in range(44, 19, -1):
        if time.time() - start_time > deadline - 20:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_result = (U.copy(), V.copy(), W.copy())
            best_rank = r
            break
    
    U_out, V_out, W_out = best_result
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
