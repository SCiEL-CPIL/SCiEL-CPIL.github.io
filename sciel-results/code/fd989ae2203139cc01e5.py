import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Seed solution (Rank 24)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)
    
    def error_function(params, r):
        # params is flattened U, V, W
        U = params[0:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        err = 0.0
        for u_idx, v_idx, w_idx in constraints:
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            err += (val - 1.0) ** 2
        return err

    def try_rank(r):
        np.random.seed(42)
        # Try a few random restarts
        for attempt in range(50):
            if time.time() - start_time > timeout - 5.0:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B
            try:
                res = minimize(error_function, params0, args=(r,), method='L-BFGS-B', options={'maxiter': 1000})
            except Exception:
                continue
                
            if res.success or res.fun < 1e-8:
                # Extract U, V, W
                U_opt = res.x[0:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify exactness
                if error_function(np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()]), r) < 1e-9:
                    return U_rounded, V_rounded, W_rounded
        
        return None

    # Try lower ranks
    for r in [18, 19, 20, 21, 22, 23]:
        if time.time() - start_time > timeout - 10.0:
            break
        sol = try_rank(r)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
