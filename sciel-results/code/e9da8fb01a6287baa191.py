import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

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
        # Try snapping to half-integers and integers
        for denom in [2, 4, 1, 3, 6]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        return False, None, None

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

    # Define objective function
    def objective(x, rank):
        U = x[:U_dim * rank].reshape(U_dim, rank)
        V = x[U_dim * rank : U_dim * rank + V_dim * rank].reshape(V_dim, rank)
        W = x[U_dim * rank + V_dim * rank :].reshape(W_dim, rank)
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    # Try to find rank < 50
    # We prioritize lower ranks but start from 49 to maximize chance of finding any improvement
    # If we find 49, we try 48, etc.
    
    ranks_to_try = list(range(49, 24, -1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try a few random seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed + target_rank * 1000)
            
            # Initialize with small random values
            # Using a scale that helps convergence
            scale = 0.5
            U_init = np.random.randn(U_dim, target_rank) * scale
            V_init = np.random.randn(V_dim, target_rank) * scale
            W_init = np.random.randn(W_dim, target_rank) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    args=(target_rank,),
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                
                if res.fun < 1e-8:
                    U_opt = res.x[:U_dim * target_rank].reshape(U_dim, target_rank)
                    V_opt = res.x[U_dim * target_rank : U_dim * target_rank + V_dim * target_rank].reshape(V_dim, target_rank)
                    W_opt = res.x[U_dim * target_rank + V_dim * target_rank:].reshape(W_dim, target_rank)
                    
                    valid, U_s, V_s, W_s = snap_and_validate(U_opt, V_opt, W_opt)
                    if valid:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_s, V_s, W_s
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            # If we found a lower rank, continue to try even lower
            continue
        else:
            # If we fail to find 49, it's unlikely we'll find 48 quickly with random restarts
            # But we can try a couple more ranks if time permits
            if target_rank == 49:
                # Try 48 and 47 with fewer seeds if time allows
                pass

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
