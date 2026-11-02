# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 24
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def check_half_integer(val, tol=1e-9):
        return abs(val * 2 - round(val * 2)) < tol
    
    def verify_solution(U, V, W, r, tol=1e-9):
        # Check half-integer constraint
        for c in range(r):
            for idx in range(nm):
                if not check_half_integer(U[idx][c]):
                    return False
            for idx in range(mp):
                if not check_half_integer(V[idx][c]):
                    return False
            for idx in range(pn):
                if not check_half_integer(W[idx][c]):
                    return False
        # Check exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def residuals(params, r):
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp : r * nm + r * mp + r * pn]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    res[idx] = s - 1.0
                    idx += 1
        return res

    # Try to find a decomposition of rank 23
    # If found, we can try lower, but 23 is the immediate target.
    # We use least_squares which is more robust than Nelder-Mead for this.
    
    ranks_to_try = [23] # Start with 23. If we have time, we could add 22, etc.
    
    for r in ranks_to_try:
        if time.time() - start_time > deadline:
            break
        
        total_vars = r * (nm + mp + pn)
        
        # Multiple attempts with different initializations
        num_attempts = 20
        for attempt in range(num_attempts):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(attempt * 1000 + r)
            x0 = np.random.uniform(-1, 1, size=total_vars)
            
            try:
                res = least_squares(residuals, x0, args=(r,), method='trf', 
                                    xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
            except Exception:
                continue
            
            if res.cost < 1e-10: # cost is 0.5 * sum(residuals^2)
                U_flat = res.x[:r * nm]
                V_flat = res.x[r * nm : r * nm + r * mp]
                W_flat = res.x[r * nm + r * mp : r * nm + r * mp + r * pn]
                
                U = U_flat.reshape(nm, r)
                V = V_flat.reshape(mp, r)
                W = W_flat.reshape(pn, r)
                
                # Round to half-integers
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                
                if verify_solution(U_list, V_list, W_list, r):
                    best_rank = r
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found solution for this rank, move to lower (if any)

    return best_result
# EVOLVE-BLOCK-END
