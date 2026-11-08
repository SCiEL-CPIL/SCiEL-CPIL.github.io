import time
import numpy as np
from scipy.optimize import minimize

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

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_and_validate(U, V, W):
        # Try multiple snap granularities
        for denom in [2, 4, 3, 6, 5, 10]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        # Also try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
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

    # Try ranks from 44 down to 38
    for target_rank in [44, 43, 42, 41, 40, 39, 38]:
        if time.time() > deadline - 15:
            break
            
        found = False
        # Try multiple random seeds with different initialization strategies
        for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 8:
                break
                
            np.random.seed(seed)
            
            # Strategy 1: Small random noise
            if seed <= 4:
                U_init = np.random.randn(U_dim, target_rank) * 0.1
                V_init = np.random.randn(V_dim, target_rank) * 0.1
                W_init = np.random.randn(W_dim, target_rank) * 0.1
            else:
                # Strategy 2: Sparse initialization with some structure
                U_init = np.zeros((U_dim, target_rank))
                V_init = np.zeros((V_dim, target_rank))
                W_init = np.zeros((W_dim, target_rank))
                
                # Randomly assign non-zero entries
                for r in range(target_rank):
                    # Pick random indices to set
                    u_idx = np.random.randint(0, U_dim, size=2)
                    v_idx = np.random.randint(0, V_dim, size=2)
                    w_idx = np.random.randint(0, W_dim, size=2)
                    
                    U_init[u_idx[0], r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    U_init[u_idx[1], r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    V_init[v_idx[0], r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    V_init[v_idx[1], r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    W_init[w_idx[0], r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    W_init[w_idx[1], r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Multi-stage optimization: coarse then fine
            try:
                # Stage 1: Coarse optimization with L-BFGS-B
                res1 = minimize(
                    objective, 
                    x0, 
                    args=(T, target_rank), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'gtol': 1e-4}
                )
                
                if not res1.success:
                    continue
                    
                # Stage 2: Fine optimization starting from stage 1 result
                res2 = minimize(
                    objective, 
                    res1.x, 
                    args=(T, target_rank), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'gtol': 1e-7}
                )
                
                # Use the best of the two stages
                best_x = res2.x if (res2.success and res2.fun < res1.fun) else res1.x
                
                U_opt = best_x[:U_dim * target_rank].reshape(U_dim, target_rank)
                V_opt = best_x[U_dim * target_rank : U_dim * target_rank + V_dim * target_rank].reshape(V_dim, target_rank)
                W_opt = best_x[U_dim * target_rank + V_dim * target_rank :].reshape(W_dim, target_rank)
                
                valid, U_s, V_s, W_s = snap_and_validate(U_opt, V_opt, W_opt)
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_s, V_s, W_s
                    found = True
                    break
            except Exception:
                pass

        if found:
            # Continue trying lower ranks if time permits
            pass

    # Return the best found
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
