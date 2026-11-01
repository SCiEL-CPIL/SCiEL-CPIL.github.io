import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_sb, V_sb, W_sb, R_sb = schoolbook()
    best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb

    # Try to find lower rank
    # Lower bound heuristic: max(n*m, m*p, p*n) is not a strict lower bound for rank,
    # but rank >= max(n*p, m*p, n*m) is often cited for some cases?
    # Actually, rank(M_{n,m,p}) >= n*p is not true.
    # Let's start from a reasonable lower bound, e.g., 20.
    
    min_rank_to_try = 20
    
    for R in range(min_rank_to_try, R_sb + 1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random values
            rng = np.random.default_rng(attempt * 1000 + R)
            U = rng.integers(-1, 2, size=(n * m, R))
            V = rng.integers(-1, 2, size=(m * p, R))
            W = rng.integers(-1, 2, size=(p * n, R))
            
            # Flatten for optimization
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                u_size = n * m * R
                v_size = m * p * R
                U = x[:u_size].reshape(n * m, R)
                V = x[u_size:u_size+v_size].reshape(m * p, R)
                W = x[u_size+v_size:].reshape(p * n, R)
                return U, V, W
            
            def residuals(x):
                U, V, W = unflatten(x)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(R):
                                s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            res.append(s - 1)
                return np.array(res)
            
            x0 = flatten(U, V, W)
            
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=10000)
            except Exception:
                continue
                
            if sol.success and sol.cost < 1e-10:
                U_sol, V_sol, W_sol = unflatten(sol.x)
                
                # Check if close to integer/half-integer
                is_good = True
                for mat in [U_sol, V_sol, W_sol]:
                    for val in mat.flatten():
                        # Check if close to integer
                        if abs(val - round(val)) > 1e-6:
                            # Check if close to half-integer
                            if abs(val * 2 - round(val * 2)) > 1e-6:
                                is_good = False
                                break
                    if not is_good:
                        break
                
                if is_good:
                    # Snap to grid
                    U_final = np.round(U_sol * 2) / 2
                    V_final = np.round(V_sol * 2) / 2
                    W_final = np.round(W_sol * 2) / 2
                    
                    # Verify exactly
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for r in range(R):
                                    s += U_final[i * m + j, r] * V_final[j * p + k, r] * W_final[k * n + i, r]
                                if abs(s - 1) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
