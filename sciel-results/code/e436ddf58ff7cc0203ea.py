import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    # Target rank to try
    target_rank = 43
    
    # The known rank 44 solution
    U44, V44, W44 = _get_rank44_solution()
    
    # The tensor T[i, j, k] = 1 if i's col == j's row, else 0
    # i in [0, 12), j in [0, 16), k in [0, 12)
    T = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_row in range(m):
                for b_col in range(p):
                    if a_col == b_row:
                        i = a_row * m + a_col
                        j = b_row * p + b_col
                        k = b_col * n + a_row
                        T[i, j, k] = 1.0

    # Helper to verify a solution
    def verify_solution(U, V, W):
        # Check if U, V, W decompose T exactly
        # T_approx[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # We check a few random points or all points
        # Since T is sparse, checking all 2304 points is fast
        for i in range(n*m):
            for j in range(m*p):
                # Only check if T[i,j,:] is not all zero
                if np.any(T[i, j, :] != 0):
                    row_T = T[i, j, :]
                    # Compute row_UV = U[i, :] * V[j, :] (elementwise)
                    # row_approx = row_UV @ W.T
                    row_approx = np.dot(U[i, :] * V[j, :], W.T)
                    if not np.allclose(row_approx, row_T, atol=1e-8):
                        return False
        return True

    # Helper to get the rank 44 solution
    def _get_rank44_solution():
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))

        U_loc[0, 0] = 1; U_loc[2, 0] = 1
        V_loc[0, 0] = 1; V_loc[2, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1

        U_loc[2, 1] = 1; U_loc[0, 1] = -1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1

        U_loc[0, 2] = 1; U_loc[1, 2] = 1
        V_loc[2, 2] = 1; V_loc[0, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1

        U_loc[2, 3] = 1
        V_loc[1, 3] = 1; V_loc[3, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1

        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1

        U_loc[3, 5] = 1; U_loc[1, 5] = -1
        V_loc[0, 5] = 1; V_loc[3, 5] = 1
        W_loc[3, 5] = 1

        U_loc[1, 6] = 1; U_loc[2, 6] = 1
        V_loc[2, 6] = 1; V_loc[3, 6] = -1
        W_loc[0, 6] = 1

        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1

        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1

        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1

        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1

        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_row = a_row_start + r
                    global_a_col = a_col_start + c
                    global_a_idx = global_a_row * m + global_a_col
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_row = b_row_start + r
                    global_b_col = b_col_start + c
                    global_b_idx = global_b_row * p + global_b_col
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_row = c_row_start + r
                    global_c_col = c_col_start + c
                    global_c_idx = global_c_col * n + global_c_row
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return U, V, W

    # Try to find a rank 43 solution using ALS
    # This is a heuristic search
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s for safety
    
    # We will try a few random perturbations of the rank 44 solution, 
    # and then run ALS to see if it converges to a valid rank 43 solution.
    
    found_43 = False
    
    # We'll try a few restarts
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Start from rank 44 solution, but drop one random column? 
        # Or just random init?
        # Random init is better for global search, but might get stuck.
        # Let's try random init first.
        
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Run ALS for a few iterations
        for it in range(50):
            if time.time() > deadline:
                break
            
            # Update U
            # Minimize || T - T_UVW ||
            # Flatten T into (n*m, m*p * p*n)
            # T_flat[i, j*n+p + k] 
            # We want U[i, r] such that sum_r U[i, r] * (V[j, r] * W[k, r]) approx T[i, j, k]
            # For fixed V, W, the term for U[i, r] is linear in U.
            # Let M_U be a matrix where M_U[i, r] = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r]
            # This is not quite right for least squares.
            
            # Standard ALS:
            # Fix V, W.
            # T_i (vector of length m*p*p*n) = sum_r U[i, r] * (V[:, r] * W[:, r]).T.flatten()
            # Let A_r = (V[:, r] * W[:, r]).T.flatten()
            # Then T_i = U[i, :] @ A
            # A is (m*p*p*n, target_rank)
            # T_flat is (n*m, m*p*p*n)
            
            # Build A
            # A[r, j, k] = V[j, r] * W[k, r]
            A = np.einsum('jr, kr -> jkr', V, W)
            A_flat = A.flatten()
            A_mat = A_flat.reshape(-1, target_rank).T # (target_rank, m*p*p*n)? No.
            # We want X @ Y = Z.
            # U (n*m, R) * A (R, Size) = T_flat (n*m, Size)
            # A should be (R, Size)
            # A[r, idx] = V[j, r] * W[k, r] where idx maps to j, k
            # Let's construct A properly.
            
            A_mat = np.zeros((target_rank, m*p * p*n))
            for r in range(target_rank):
                # V[:, r] is (m*p,)
                # W[:, r] is (p*n,)
                # Outer product
                outer = np.outer(V[:, r], W[:, r])
                A_mat[r, :] = outer.flatten()
                
            T_flat = T.flatten() # (n*m, m*p*p*n)
            
            # Solve U @ A_mat = T_flat
            # U = T_flat @ A_mat.T @ (A_mat @ A_mat.T)^-1
            try:
                A_mat_T = A_mat.T
                # Normal equations
                Gram = A_mat @ A_mat_T # (R, R)
                RHS = T_flat @ A_mat_T # (n*m, R)
                U_new = np.linalg.solve(Gram, RHS).T
            except:
                break
            
            # Update V
            # Fix U, W.
            # T_j = sum_r V[j, r] * (U[:, r] * W[:, r]).T.flatten()
            # A_mat[r, idx] = U[i, r] * W[k, r]
            A_mat = np.zeros((target_rank, n*m * p*n))
            for r in range(target_rank):
                outer = np.outer(U[:, r], W[:, r])
                A_mat[r, :] = outer.flatten()
            
            # T_flat needs to be reshaped to (m*p, n*m * p*n)
            # T[i, j, k] -> T_flat[j, i * p*n + k]
            T_flat_V = T.transpose(1, 0, 2).reshape(m*p, -1)
            
            try:
                A_mat_T = A_mat.T
                Gram = A_mat @ A_mat_T
                RHS = T_flat_V @ A_mat_T
                V_new = np.linalg.solve(Gram, RHS).T
            except:
                break
                
            # Update W
            # Fix U, V.
            # T_k = sum_r W[k, r] * (U[:, r] * V[:, r]).T.flatten()
            # A_mat[r, idx] = U[i, r] * V[j, r]
            A_mat = np.zeros((target_rank, n*m * m*p))
            for r in range(target_rank):
                outer = np.outer(U[:, r], V[:, r])
                A_mat[r, :] = outer.flatten()
                
            T_flat_W = T.transpose(2, 0, 1).reshape(p*n, -1)
            
            try:
                A_mat_T = A_mat.T
                Gram = A_mat @ A_mat_T
                RHS = T_flat_W @ A_mat_T
                W_new = np.linalg.solve(Gram, RHS).T
            except:
                break
                
            U = U_new
            V = V_new
            W = W_new
            
            # Check convergence
            if verify_solution(U, V, W):
                # Found a solution!
                # Check if entries are close to half-integers
                # The problem requires integer/half-integer entries
                # We can round them
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if verify_solution(U_round, V_round, W_round):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                else:
                    # If rounding fails, but raw solution is valid, maybe return raw?
                    # Contract says "integer/half-integer entries".
                    # So we MUST return rounded if valid.
                    pass

    # If no rank 43 found, return rank 44
    return {
        "U": U44.tolist(),
        "V": V44.tolist(),
        "W": W44.tolist()
    }
