import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 100.0  # Leave margin for 150s limit
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    nm, mp, pn = n * m, m * p, p * n
    
    def error_func(params, r):
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp :]
        
        U = U_flat.reshape(r, nm)
        V = V_flat.reshape(r, mp)
        W = W_flat.reshape(r, pn)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx] * V[c, v_idx] * W[c, w_idx]
                    err += (s - 1.0) ** 2
        return err

    def check_solution(U, V, W, r):
        for c in range(r):
            for val in U[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in V[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in W[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx] * V[c, v_idx] * W[c, w_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_and_return(U, V, W, r):
        U_list = [[round(val * 2) / 2.0 for val in row] for row in U.tolist()]
        V_list = [[round(val * 2) / 2.0 for val in row] for row in V.tolist()]
        W_list = [[round(val * 2) / 2.0 for val in row] for row in W.tolist()]
        if check_solution(np.array(U_list), np.array(V_list), np.array(W_list), r):
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    # Try ranks from 23 down to 18
    for r in range(rank_sb - 1, 17, -1):
        if time.time() - start_time > max_time:
            break
            
        total_vars = r * (nm + mp + pn)
        max_attempts = 3 if r >= 22 else 5
        
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with schoolbook-like structure plus noise
            x0 = np.zeros(total_vars)
            # Copy schoolbook pattern for first min(r, 24) terms
            for c in range(min(r, rank_sb)):
                i, jk = divmod(c, m * p)
                j, k = divmod(jk, p)
                x0[c * nm + i * m + j] = 1.0
                x0[r * nm + c * mp + j * p + k] = 1.0
                x0[r * nm + r * mp + c * pn + k * n + i] = 1.0
            # Add small noise to remaining
            if r < rank_sb:
                pass  # already zero
            
            from scipy.optimize import minimize
            try:
                res = minimize(error_func, x0, args=(r,), method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                U_flat = res.x[:r * nm]
                V_flat = res.x[r * nm : r * nm + r * mp]
                W_flat = res.x[r * nm + r * mp :]
                
                U = U_flat.reshape(r, nm)
                V = V_flat.reshape(r, mp)
                W = W_flat.reshape(r, pn)
                
                result = verify_and_return(U, V, W, r)
                if result is not None:
                    return result
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
