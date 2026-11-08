import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-8):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_and_validate(U, V, W):
        # Try multiple snap granularities
        for denom in [1, 2, 3, 4, 5, 6, 10, 12]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        return False, None, None

    def objective(x, T, rank):
        U = x[:U_dim * rank].reshape(U_dim, rank)
        V = x[U_dim * rank : U_dim * rank + V_dim * rank].reshape(V_dim, rank)
        W = x[U_dim * rank + V_dim * rank :].reshape(W_dim, rank)
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def get_trivial_solution():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U50, V50, W50 = get_trivial_solution()
    
    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Strategy: Random Sparse V + Optimization
    # We try to find a solution with rank < 50.
    # We fix V to be sparse (2 non-zeros per row) to reduce the effective degrees of freedom
    # and guide the optimizer towards a structured solution.
    
    for target_rank in range(30, 50):
        if time.time() > deadline - 20:
            break
        
        if target_rank >= best_rank:
            continue
            
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(seed + target_rank)
            
            # Generate sparse V: 25 x target_rank
            # Each row has exactly 2 non-zeros (1 or -1)
            V_init = np.zeros((V_dim, target_rank))
            for b in range(V_dim):
                # Pick 2 random columns
                cols = np.random.choice(target_rank, 2, replace=False)
                V_init[b, cols[0]] = 1
                V_init[b, cols[1]] = np.random.choice([-1, 1])
            
            # Initialize U and W randomly with small integers
            U_init = np.random.choice([-1, 0, 1], size=(U_dim, target_rank))
            W_init = np.random.choice([-1, 0, 1], size=(W_dim, target_rank))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Optimization
                res = minimize(
                    objective, 
                    x0, 
                    args=(T, target_rank), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-10}
                )
                
                U_opt = res.x[:U_dim * target_rank].reshape(U_dim, target_rank)
                V_opt = res.x[U_dim * target_rank : U_dim * target_rank + V_dim * target_rank].reshape(V_dim, target_rank)
                W_opt = res.x[U_dim * target_rank + V_dim * target_rank :].reshape(W_dim, target_rank)
                
                valid, U_s, V_s, W_s = snap_and_validate(U_opt, V_opt, W_opt)
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_s, V_s, W_s
                    found = True
                    break
            except Exception:
                pass

        if found:
            # If we found a solution at this rank, we can try to go lower
            # But we are iterating from low to high, so we should break if we found the minimal?
            # No, we want the MINIMAL rank.
            # If we found rank 30, we should check if 29 is possible?
            # The loop goes 30, 31, ...
            # If we find 30, we should break and return 30?
            # Yes, because we are looking for the minimum.
            # However, it's possible 30 is not the minimum, but we found it.
            # We should continue searching lower ranks if possible, but we started at 30.
            # So if we find 30, it is the best so far.
            # We can try to go lower by restarting the loop with a lower start?
            # For now, break and return.
            break

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
