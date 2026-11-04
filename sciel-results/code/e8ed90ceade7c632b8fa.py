import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook solution
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school

    # Try to find a lower rank
    # Theoretical lower bound is roughly nmp / (some factor)
    # For 2x3x3, try R=17, 16, ...
    
    def solve_U(V, W, R):
        """
        Solve for U given V, W.
        U has shape (n*m, R).
        For each (i, j), we solve a linear system for U[i*m+j, :].
        """
        U_new = np.zeros((n*m, R))
        # Precompute V and W flattened? No, they are already matrices.
        # V is (m*p, R), W is (p*n, R).
        
        for i in range(n):
            for j in range(m):
                # Build A and b for this (i, j)
                # Equations indexed by (j', k, i', k')
                # j' in [0, m), k in [0, p), i' in [0, n), k' in [0, p)
                # Total equations: m * p * n * p
                n_eq = m * p * n * p
                A_mat = np.zeros((n_eq, R))
                b_vec = np.zeros(n_eq)
                
                eq_idx = 0
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                # Target: delta_{i, ip} * delta_{j, jp} * delta_{k, kp}
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                b_vec[eq_idx] = target
                                
                                for r in range(R):
                                    # Coefficient of U[i*m+j, r] is V[jp*p+k, r] * W[kp*n+ip, r]
                                    A_mat[eq_idx, r] = V[jp*p+k, r] * W[kp*n+ip, r]
                                eq_idx += 1
                                
                # Solve A_mat u = b_vec
                u_sol, res, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U_new[i*m+j, :] = u_sol
        return U_new

    def solve_V(U, W, R):
        """
        Solve for V given U, W.
        V has shape (m*p, R).
        For each (j', k), we solve for V[j'*p+k, :].
        """
        V_new = np.zeros((m*p, R))
        
        for jp in range(m):
            for k in range(p):
                n_eq = n * m * n * p
                A_mat = np.zeros((n_eq, R))
                b_vec = np.zeros(n_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for ip in range(n):
                            for kp in range(p):
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                b_vec[eq_idx] = target
                                
                                for r in range(R):
                                    A_mat[eq_idx, r] = U[i*m+j, r] * W[kp*n+ip, r]
                                eq_idx += 1
                                
                v_sol, res, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V_new[jp*p+k, :] = v_sol
        return V_new

    def solve_W(U, V, R):
        """
        Solve for W given U, V.
        W has shape (p*n, R).
        For each (k', i'), we solve for W[k'*n+i', :].
        """
        W_new = np.zeros((p*n, R))
        
        for kp in range(p):
            for ip in range(n):
                n_eq = n * m * m * p
                A_mat = np.zeros((n_eq, R))
                b_vec = np.zeros(n_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                b_vec[eq_idx] = target
                                
                                for r in range(R):
                                    A_mat[eq_idx, r] = U[i*m+j, r] * V[jp*p+k, r]
                                eq_idx += 1
                                
                w_sol, res, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_new[kp*n+ip, :] = w_sol
        return W_new

    def compute_residual(U, V, W, R):
        """
        Compute the Frobenius norm of the error.
        """
        # We only need to check the 18 non-zero targets?
        # No, we need to ensure the 0s are also 0.
        # But the 0s are usually satisfied if the 1s are, due to symmetry?
        # No, we must check all.
        # However, for speed, we can check a subset or use the fact that 
        # if the system is solved well, the residual is small.
        
        # Let's check all 324 entries.
        total_res = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                val = 0.0
                                for r in range(R):
                                    val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                                total_res += (val - target)**2
        return total_res

    # Search for lower ranks
    for target_rank in range(rank_school - 1, max(rank_school - 4, 10), -1):
        if time.time() > deadline:
            break
            
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + trial)
            U_t = np.random.randn(n*m, target_rank) * 0.5
            V_t = np.random.randn(m*p, target_rank) * 0.5
            W_t = np.random.randn(p*n, target_rank) * 0.5
            
            # ALS
            converged = False
            for iteration in range(100):
                if time.time() > deadline:
                    break
                    
                U_t = solve_U(V_t, W_t, target_rank)
                V_t = solve_V(U_t, W_t, target_rank)
                W_t = solve_W(U_t, V_t, target_rank)
                
                # Check convergence
                res = compute_residual(U_t, V_t, W_t, target_rank)
                if res < 1e-8:
                    converged = True
                    break
                
                if time.time() > deadline:
                    break
            
            if converged:
                # Verify and round
                # Try to round to nearest 0.5 or integer
                U_r = np.round(U_t * 2) / 2
                V_r = np.round(V_t * 2) / 2
                W_r = np.round(W_t * 2) / 2
                
                res_r = compute_residual(U_r, V_r, W_r, target_rank)
                if res_r < 1e-6:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    # Convert to list
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
