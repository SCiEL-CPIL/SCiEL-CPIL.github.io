import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_schoolbook, V_schoolbook, W_schoolbook
    best_rank = rank_schoolbook
    
    # Build constraint matrices for alternating minimization
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Precompute index mappings
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i, j, k))
    
    num_constraints = len(constraints)
    
    def build_A_for_W(U, V, target_rank):
        """Build linear system A_w * W_flat = b_w"""
        A_w = np.zeros((num_constraints, p * n * target_rank))
        for eq_idx, (i, j, k) in enumerate(constraints):
            for r in range(target_rank):
                A_w[eq_idx, (k * n + i) * target_rank + r] = U[i * m + j, r] * V[j * p + k, r]
        return A_w, np.ones(num_constraints)
    
    def build_A_for_U(V, W, target_rank):
        """Build linear system A_u * U_flat = b_u"""
        A_u = np.zeros((num_constraints, n * m * target_rank))
        for eq_idx, (i, j, k) in enumerate(constraints):
            for r in range(target_rank):
                A_u[eq_idx, (i * m + j) * target_rank + r] = V[j * p + k, r] * W[k * n + i, r]
        return A_u, np.ones(num_constraints)
    
    def build_A_for_V(U, W, target_rank):
        """Build linear system A_v * V_flat = b_v"""
        A_v = np.zeros((num_constraints, m * p * target_rank))
        for eq_idx, (i, j, k) in enumerate(constraints):
            for r in range(target_rank):
                A_v[eq_idx, (j * p + k) * target_rank + r] = U[i * m + j, r] * W[k * n + i, r]
        return A_v, np.ones(num_constraints)
    
    def compute_error(U, V, W):
        """Compute sum of squared errors"""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    error += (val - 1.0) ** 2
        return error
    
    def try_round_and_verify(U, V, W, tolerance=1e-8):
        """Try rounding to half-integers and verify"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        error_r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                    error_r += (val - 1.0) ** 2
        
        if error_r < tolerance:
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    # Try to find lower-rank solutions using alternating minimization
    # Try ranks from 65 to 71
    for target_rank in range(65, 72):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random initializations
        for seed in range(15):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed + target_rank * 1000)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            prev_error = float('inf')
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Solve for W
                A_w, b_w = build_A_for_W(U, V, target_rank)
                w_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = w_sol.reshape(p * n, target_rank)
                
                # Solve for U
                A_u, b_u = build_A_for_U(V, W, target_rank)
                u_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = u_sol.reshape(n * m, target_rank)
                
                # Solve for V
                A_v, b_v = build_A_for_V(U, W, target_rank)
                v_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = v_sol.reshape(m * p, target_rank)
                
                # Check convergence
                error = compute_error(U, V, W)
                
                if error < 1e-12:
                    # Try rounding
                    U_r, V_r, W_r, success = try_round_and_verify(U, V, W)
                    if success:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        found = True
                        break
                
                # Check for convergence stagnation
                if abs(prev_error - error) < 1e-15:
                    break
                prev_error = error
            
            if found:
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
