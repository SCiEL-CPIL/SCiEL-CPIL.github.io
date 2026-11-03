"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    # Build the target tensor T
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    target_set = set(target_indices)
    T_target = np.zeros((n * m, m * p, p * n))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    # Function to check validity and snap to half-integers
    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        
        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None
            
        # Convert back to lists
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]
        
        return {"U": U_out, "V": V_out, "W": W_out}

    # Optimization function
    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        def loss(params):
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            
            U_p = U_flat.reshape((n * m), r)
            V_p = V_flat.reshape((m * p), r)
            W_p = W_flat.reshape((p * n), r)
            
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        from scipy.optimize import minimize
        
        try:
            # Use L-BFGS-B with more iterations allowed
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 1000})
        except Exception:
            return None
            
        # Check if the solution is good enough
        if res.fun < 1e-6:
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_sol = res.x[:size_U].reshape((n * m), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
            W_sol = res.x[size_U + size_V:].reshape((p * n), r)
            
            result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
            if result:
                return result
        return None

    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank
    # Try ranks in descending order
    ranks_to_try = [45, 44, 43, 42, 41, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        found = False
        for seed in [42, 123, 999, 111, 222]:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result:
                best_result = result
                best_rank = r
                found = True
                break
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
