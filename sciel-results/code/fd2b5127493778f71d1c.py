import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = get_target_tensor()
    
    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def verify(U, V, W):
        T = compute_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-8)
    
    def try_round_and_verify(U_num, V_num, W_num, denom=2):
        """Try rounding to multiples of 1/denom and verify."""
        U_rnd = np.round(U_num * denom) / denom
        V_rnd = np.round(V_num * denom) / denom
        W_rnd = np.round(W_num * denom) / denom
        if verify(U_rnd, V_rnd, W_rnd):
            return U_rnd, V_rnd, W_rnd, True
        return None, None, None, False
    
    np.random.seed(42)
    
    found = False
    
    # Try ranks from 10 upward
    for target_rank in [10, 11]:
        if found or time.time() > deadline - 10.0:
            break
        
        def objective(params, tr=target_rank):
            if time.time() > deadline:
                return 1e9
            nm = n * m
            mp = m * p
            pn = p * n
            U_flat = params[:nm*tr]
            V_flat = params[nm*tr:nm*tr + mp*tr]
            W_flat = params[nm*tr + mp*tr:]
            
            U = U_flat.reshape(nm, tr)
            V = V_flat.reshape(mp, tr)
            W = W_flat.reshape(pn, tr)
            
            T_prime = compute_tensor(U, V, W)
            diff = T_prime - T_target
            return np.sum(diff**2)
        
        # Strategy 1: Use schoolbook columns as initial guess
        if target_rank <= rank_school:
            U_init = U_sb[:, :target_rank].copy()
            V_init = V_sb[:, :target_rank].copy()
            W_init = W_sb[:, :target_rank].copy()
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, params0, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < 1e-10:
                    U_flat = res.x[:n*m*target_rank]
                    V_flat = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = res.x[n*m*target_rank + m*p*target_rank:]
                    U_num = U_flat.reshape(n*m, target_rank)
                    V_num = V_flat.reshape(m*p, target_rank)
                    W_num = W_flat.reshape(p*n, target_rank)
                    
                    # Try different rounding denominators
                    for denom in [1, 2, 4, 8]:
                        U_rnd, V_rnd, W_rnd, ok = try_round_and_verify(U_num, V_num, W_num, denom)
                        if ok:
                            best_U, best_V, best_W = U_rnd.tolist(), V_rnd.tolist(), W_rnd.tolist()
                            best_rank = target_rank
                            found = True
                            break
                        if time.time() > deadline - 5.0:
                            break
                    if found:
                        break
            except Exception:
                pass
        
        # Strategy 2: Random initializations
        if not found:
            max_attempts = 30 if target_rank == 10 else 20
            for attempt in range(max_attempts):
                if time.time() > deadline - 5.0:
                    break
                
                scale = 0.3 + 0.2 * np.random.rand()
                params0 = np.random.randn((n*m + m*p + p*n) * target_rank) * scale
                try:
                    res = minimize(objective, params0, method='L-BFGS-B',
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                    if res.fun < 1e-8:
                        U_flat = res.x[:n*m*target_rank]
                        V_flat = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                        W_flat = res.x[n*m*target_rank + m*p*target_rank:]
                        U_num = U_flat.reshape(n*m, target_rank)
                        V_num = V_flat.reshape(m*p, target_rank)
                        W_num = W_flat.reshape(p*n, target_rank)
                        
                        for denom in [1, 2, 4, 8]:
                            U_rnd, V_rnd, W_rnd, ok = try_round_and_verify(U_num, V_num, W_num, denom)
                            if ok:
                                best_U, best_V, best_W = U_rnd.tolist(), V_rnd.tolist(), W_rnd.tolist()
                                best_rank = target_rank
                                found = True
                                break
                            if time.time() > deadline - 3.0:
                                break
                        if found:
                            break
                except Exception:
                    continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
