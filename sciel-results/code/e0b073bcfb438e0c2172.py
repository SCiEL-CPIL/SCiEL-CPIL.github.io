import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Target: Find rank minimal decomposition for n x m * m x p
    # Lower bound: max(n*m, m*p, p*n)
    lb = max(n*m, m*p, p*n)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    def make_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T

    def verify(U, V, W, T, tol=1e-6):
        R = U.shape[1]
        try:
            Temp = np.einsum('ur,vr->uvr', U, V)
            T_approx = np.einsum('uvr,wr->uvw', Temp, W)
            return np.allclose(T_approx, T, atol=tol)
        except:
            return False

    def try_find(n, m, p, target_rank, time_limit):
        T = make_tensor(n, m, p)
        nm, mp, pn = n*m, m*p, p*n
        start_time = time.time()
        
        # Try several random starts
        max_starts = 50
        for s in range(max_starts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with small integers/floats
            U = np.random.uniform(-1, 1, (nm, target_rank))
            V = np.random.uniform(-1, 1, (mp, target_rank))
            W = np.random.uniform(-1, 1, (pn, target_rank))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_x = x[:nm*target_rank].reshape(nm, target_rank)
                V_x = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_x = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                try:
                    Temp = np.einsum('ur,vr->uvr', U_x, V_x)
                    T_approx = np.einsum('uvr,wr->uvw', Temp, W_x)
                    err = np.sum((T_approx - T)**2)
                    return err
                except:
                    return 1e10

            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-6:
                U_opt = res.x[:nm*target_rank].reshape(nm, target_rank)
                V_opt = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_opt = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                # Try to snap to integers or half-integers
                for factor in [1, 2]:
                    U_round = np.round(U_opt * factor) / factor
                    V_round = np.round(V_opt * factor) / factor
                    W_round = np.round(W_opt * factor) / factor
                    if verify(U_round, V_round, W_round, T, tol=1e-6):
                        return U_round, V_round, W_round
                
                # If rounding fails, check if the float solution is close enough
                if verify(U_opt, V_opt, W_opt, T, tol=1e-5):
                    return U_opt, V_opt, W_opt
        
        return None

    time_limit_total = 140 # Leave 10s for safety
    t_start = time.time()
    
    # Step 1: Try to find rank 12 for 2x3x3
    # If found, we can construct rank 24 for 2x3x6
    res_sub = try_find(2, 3, 3, 12, 60)
    if res_sub:
        U_sub, V_sub, W_sub = res_sub
        R = 12
        U_new = np.hstack([U_sub, U_sub])
        V_new = np.zeros((18, 2*R))
        W_new = np.zeros((12, 2*R))
        
        V_new[0:9, 0:R] = V_sub
        V_new[9:18, R:2*R] = V_sub
        
        W_new[0:6, 0:R] = W_sub
        W_new[6:12, R:2*R] = W_sub
        
        T_full = make_tensor(2, 3, 6)
        if verify(U_new, V_new, W_new, T_full, tol=1e-6):
            return {
                "U": U_new.tolist(),
                "V": V_new.tolist(),
                "W": W_new.tolist()
            }
            
    # Step 2: If sub-problem failed, try direct search for 2x3x6 with rank 18, 20, 22, 24
    # We know rank 24 is possible if we find rank 12 sub-problem, but if that failed,
    # maybe we can find a direct rank 24?
    remaining_time = time_limit_total - (time.time() - t_start)
    if remaining_time > 10:
        for R in [24, 22, 20, 18]:
            if time.time() - t_start > time_limit_total - 5:
                break
            res_direct = try_find(2, 3, 6, R, min(10, remaining_time - 5))
            if res_direct:
                U_d, V_d, W_d = res_direct
                return {
                    "U": U_d.tolist(),
                    "V": V_d.tolist(),
                    "W": W_d.tolist()
                }
            remaining_time = time_limit_total - (time.time() - t_start)
            if remaining_time < 10:
                break

    # Fallback: Schoolbook
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
