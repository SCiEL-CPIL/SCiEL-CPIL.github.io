import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 120.0  # Leave margin for 150s limit
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((rank_sb, nm))
    V_sb = np.zeros((rank_sb, mp))
    W_sb = np.zeros((rank_sb, pn))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[col, i * m + j] = 1
                V_sb[col, j * p + k] = 1
                W_sb[col, k * n + i] = 1
                col += 1

    def check_solution(U, V, W, r):
        # Check if entries are half-integers
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
        
        # Check if it decomposes the tensor
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

    # Try rank 23
    r = 23
    total_vars = r * (nm + mp + pn)
    
    def residuals(params):
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp :]
        
        U = U_flat.reshape(r, nm)
        V = V_flat.reshape(r, mp)
        W = W_flat.reshape(r, pn)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx] * V[c, v_idx] * W[c, w_idx]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    bounds = (-10.0, 10.0)
    
    # Try to find a solution
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization
        x0 = np.random.uniform(-1, 1, total_vars)
        
        try:
            res = least_squares(residuals, x0, bounds=bounds, method='trf', max_nfev=2000)
        except Exception:
            continue
            
        if res.cost < 1e-12:
            U_flat = res.x[:r * nm]
            V_flat = res.x[r * nm : r * nm + r * mp]
            W_flat = res.x[r * nm + r * mp :]
            
            U = U_flat.reshape(r, nm)
            V = V_flat.reshape(r, mp)
            W = W_flat.reshape(r, pn)
            
            result = verify_and_return(U, V, W, r)
            if result is not None:
                return result

    # If rank 23 not found, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
