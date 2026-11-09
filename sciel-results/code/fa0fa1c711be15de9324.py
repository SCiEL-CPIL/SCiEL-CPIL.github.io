import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Construct the target tensor
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Schoolbook rank
    rank_school = n * m * p
    
    # Time guard
    start_time = time.time()
    max_time = 130  # Leave 20s for safety against 150s limit
    
    # Initial solution: Schoolbook
    U_init = np.zeros((dim_U, rank_school))
    V_init = np.zeros((dim_V, rank_school))
    W_init = np.zeros((dim_W, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_init[u_idx, col] = 1
                V_init[v_idx, col] = 1
                W_init[w_idx, col] = 1
                col += 1
    best_U, best_V, best_W = U_init, V_init, W_init

    def error_func(params, k):
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec - T) ** 2)
        return err

    # Try ranks from a reasonable lower bound up to rank_school - 1
    # Known lower bounds are around 20-25 for 3x3x4
    # We start at 20 to be safe, but 23 is often cited. Let's start at 20 to allow more search time.
    # Actually, let's start at 23 as in the original, but allow more restarts and better init.
    
    # We will try ranks from 23 upwards
    for k in range(23, rank_school):
        if time.time() - start_time > max_time:
            break
            
        # Increase restarts to improve chance of finding a solution
        # 5 restarts per rank
        found = False
        for restart in range(5):
            if time.time() - start_time > max_time:
                break
                
            # Use a deterministic seed based on k and restart for reproducibility
            np.random.seed(1000 + k * 10 + restart)
            
            # Initialization strategy:
            # Random init with scaling
            # Scale factors: U, V, W entries should be O(1)
            # Since T entries are 0 or 1, and sum_c U V W = T,
            # if k is ~25, each term is ~1/k, so U,V,W ~ k^(1/3) ~ 3?
            # Or if sparse, larger.
            # Let's try scaling by 1.0
            scale = 1.0
            init_U = np.random.randn(dim_U, k) * scale
            init_V = np.random.randn(dim_V, k) * scale
            init_W = np.random.randn(dim_W, k) * scale
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                # Use L-BFGS-B
                # maxiter increased to 200
                res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                # Extract best params
                p_best = res.x
                U_opt = p_best[:dim_U * k].reshape((dim_U, k))
                V_opt = p_best[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W_opt = p_best[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                # Round to nearest integer or half-integer
                # The task allows integer/half-integer entries.
                # Rounding to 0.5 precision
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                Rec_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                if np.allclose(Rec_round, T, atol=1e-9):
                    # Success
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    
            except Exception:
                pass

    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
