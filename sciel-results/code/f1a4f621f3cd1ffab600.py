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
    
    # Precompute index arrays for vectorized error computation
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    u_idx = (i_vals * m + j_vals).flatten()
    v_idx = (j_vals * p + k_vals).flatten()
    w_idx = (k_vals * n + i_vals).flatten()
    num_terms = n * m * p
    
    def error_func(params, r):
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp :]
        
        U = U_flat.reshape(r, nm)
        V = V_flat.reshape(r, mp)
        W = W_flat.reshape(r, pn)
        
        # Vectorized: for each term, compute sum over rank
        # U[u_idx, :], V[v_idx, :], W[w_idx, :] are (num_terms, r)
        U_sel = U[:, u_idx].T  # (num_terms, r)
        V_sel = V[:, v_idx].T  # (num_terms, r)
        W_sel = W[:, w_idx].T  # (num_terms, r)
        
        # Product along rank axis
        prod = U_sel * V_sel * W_sel  # (num_terms, r)
        sums = prod.sum(axis=1)  # (num_terms,)
        
        err = np.sum((sums - 1.0) ** 2)
        return float(err)
    
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
                    u_idx_val = i * m + j
                    v_idx_val = j * p + k
                    w_idx_val = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx_val] * V[c, v_idx_val] * W[c, w_idx_val]
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

    def polish(x0, r, max_iter=5000):
        """Two-stage polish: first coarse, then fine with tighter tolerance."""
        from scipy.optimize import minimize
        
        # Stage 1: Coarse optimization
        try:
            res1 = minimize(error_func, x0, args=(r,), method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
        except Exception:
            return None
        
        if res1.fun > 1e-6:
            return None
        
        # Stage 2: Fine optimization from stage 1 result
        try:
            res2 = minimize(error_func, res1.x, args=(r,), method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        except Exception:
            return res1.x if res1.fun < 1e-8 else None
        
        return res2.x if res2.fun < 1e-8 else (res1.x if res1.fun < 1e-8 else None)

    # Try ranks from 23 down to 18
    for r in range(rank_sb - 1, 17, -1):
        if time.time() - start_time > max_time:
            break
            
        total_vars = r * (nm + mp + pn)
        max_attempts = 4 if r >= 22 else 6
        
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with schoolbook-like structure
            x0 = np.zeros(total_vars)
            for c in range(min(r, rank_sb)):
                i, jk = divmod(c, m * p)
                j, k = divmod(jk, p)
                x0[c * nm + i * m + j] = 1.0
                x0[r * nm + c * mp + j * p + k] = 1.0
                x0[r * nm + r * mp + c * pn + k * n + i] = 1.0
            
            # Add noise for diversity in attempts > 0
            if attempt > 0:
                noise_scale = 0.1 * attempt
                x0 += noise_scale * np.random.randn(total_vars)
            
            x_result = polish(x0, r)
            if x_result is not None:
                U_flat = x_result[:r * nm]
                V_flat = x_result[r * nm : r * nm + r * mp]
                W_flat = x_result[r * nm + r * mp :]
                
                U = U_flat.reshape(r, nm)
                V = V_flat.reshape(r, mp)
                W = W_flat.reshape(r, pn)
                
                result = verify_and_return(U, V, W, r)
                if result is not None:
                    return result
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
