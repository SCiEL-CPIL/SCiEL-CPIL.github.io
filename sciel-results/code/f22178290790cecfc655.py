import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor T
    # T[u_idx, v_idx, w_idx] = 1.0
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

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

    def calculate_residual(x, rank):
        U = x[:U_dim * rank].reshape(U_dim, rank)
        V = x[U_dim * rank : U_dim * rank + V_dim * rank].reshape(V_dim, rank)
        W = x[U_dim * rank + V_dim * rank :].reshape(W_dim, rank)
        # Compute product tensor
        # P[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        P = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = P - T
        return np.sum(diff * diff)

    def snap_and_validate(U, V, W, atol=1e-5):
        # Try snapping to half-integers, integers, and other simple rationals
        # We check if the snapped version exactly reproduces T
        candidates = []
        
        # Helper to check a set of matrices
        def check(U_s, V_s, W_s):
            P = np.einsum('ar,br,cr->abc', U_s, V_s, W_s, optimize=True)
            return np.allclose(P, T, atol=atol)

        # 1. Integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check(U_i, V_i, W_i):
            return True, U_i, V_i, W_i

        # 2. Half-integers
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        if check(U_h, V_h, W_h):
            return True, U_h, V_h, W_h

        # 3. Try to snap to 0, 1, -1 if close
        # This is often the case for these decompositions
        def snap_simple(M):
            M_s = M.copy()
            M_s[np.abs(M_s) < 1e-4] = 0
            M_s[np.abs(M_s - 1) < 1e-4] = 1
            M_s[np.abs(M_s + 1) < 1e-4] = -1
            M_s[np.abs(M_s - 0.5) < 1e-4] = 0.5
            M_s[np.abs(M_s + 0.5) < 1e-4] = -0.5
            return M_s

        U_s = snap_simple(U)
        V_s = snap_simple(V)
        W_s = snap_simple(W)
        if check(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
            
        return False, None, None

    # Search for lower ranks
    # Theoretical lower bound is not trivially known, but < 50 is likely possible.
    # We try ranks from 45 down to 35.
    ranks_to_try = list(range(45, 34, -1))
    
    # To make it faster, we limit the number of seeds per rank
    max_seeds_per_rank = 5
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(max_seeds_per_rank):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed + target_rank * 100)
            
            # Initialization: Small random values
            scale = 0.1
            U_init = np.random.randn(U_dim, target_rank) * scale
            V_init = np.random.randn(V_dim, target_rank) * scale
            W_init = np.random.randn(W_dim, target_rank) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimization
            try:
                # Use L-BFGS-B
                res = minimize(
                    calculate_residual, 
                    x0, 
                    args=(target_rank,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                
                if res.fun < 1e-6: # Close to zero
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
            # If we found a valid solution, we can continue to try lower ranks
            # but we must ensure we don't exceed time.
            pass

    # Return the best found
    # Ensure lists are of floats
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
