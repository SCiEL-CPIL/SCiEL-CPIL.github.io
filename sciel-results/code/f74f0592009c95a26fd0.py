import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def get_current_solution():
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10)
        U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
        U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
        U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_solution(sol, max_tests=10):
        U = np.array(sol['U'])
        V = np.array(sol['V'])
        W = np.array(sol['W'])
        rank = U.shape[1]
        
        T = np.zeros((12, 16, 12))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    T[i, j, k] = 1.0
        
        for _ in range(max_tests):
            i = np.random.randint(0, 12)
            j = np.random.randint(0, 16)
            k = np.random.randint(0, 12)
            computed = np.sum(U[i, :] * V[j, :] * W[k, :])
            expected = T[i, j, k]
            if abs(computed - expected) > 1e-9:
                return False
        return True
    
    def try_sparse_search(best_rank=44, max_rank=40):
        """
        Search for a lower-rank decomposition using sparse linear regression.
        
        The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        
        We parameterize each rank-1 term by its support sets:
        - U term has support S_U subset of {0,...,11}
        - V term has support S_V subset of {0,...,15}  
        - W term has support S_W subset of {0,...,11}
        
        For binary supports with coefficients +/-1, a rank-1 term contributes:
        T[i,j,k] += u_i * v_j * w_k where u_i in {-1,0,1}, etc.
        
        We'll enumerate candidate rank-1 terms with small supports and solve
        for a sparse combination.
        """
        if time.time() > deadline - 20:
            return None
        
        # Build the target tensor
        T = np.zeros((12, 16, 12))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    T[i, j, k] = 1.0
        
        T_flat = T.flatten()
        
        # Generate candidate rank-1 terms with small supports
        # Each candidate is (U_vec, V_vec, W_vec) where each has small support
        # and entries in {-1, 0, 1}
        
        candidates = []
        
        # Strategy: generate candidates from "structured" patterns
        # Focus on terms that have support in specific regions
        
        # Generate all possible (S_U, S_V, S_W) with |S_U|*|S_V|*|S_W| <= some bound
        # But that's too many. Instead, use heuristics.
        
        # For each possible pattern of which (i,j,k) triples get contribution,
        # try to find a rank-1 decomposition
        
        # Simpler approach: use the known structure of matrix multiplication
        # The tensor T[i,j,k] is 1 when i = (row_a, col_a), j = (col_a, col_c), k = (col_c, row_a)
        # So the "natural" rank-1 terms are indexed by (row_a, col_a, col_c)
        
        # But we can combine multiple such terms into one rank-1 term if their
        # supports are compatible
        
        # Let's try a different approach: assume the decomposition has rank R,
        # and use a greedy/iterative method to find the terms
        
        # For now, let's try to verify if rank 43 is achievable by checking
        # if we can remove one term from the rank-44 solution and adjust others
        
        # Actually, let's try a more systematic search
        # We'll use the fact that the tensor is sparse (48 non-zero entries out of 12*16*12=2304)
        
        # Key insight: each rank-1 term with binary entries can contribute to at most
        # |S_U| * |S_V| * |S_W| entries of T
        
        # Let's generate candidates based on the structure
        # For each row_a in {0,1,2}, col_a in {0,1,2,3}, col_c in {0,1,2,3}
        # The "base" term has:
        # U support: {row_a * 4 + col_a}
        # V support: {col_a * 4 + col_c}
        # W support: {col_c * 3 + row_a}
        
        # But we can combine terms that share structure
        
        # Let's try: for each pair of col_a values, can we combine?
        # This is getting complex. Let's use a numerical approach instead.
        
        # Approach: Assume rank R. We have R unknown vectors u_r, v_r, w_r.
        # The constraint is: sum_r u_r[i] * v_r[j] * w_r[k] = T[i,j,k] for all i,j,k
        # This is a system of 2304 equations in 3*R*(12+16+12) = 3*R*40 = 120R unknowns
        # (if we allow real-valued entries)
        
        # But we want integer/half-integer entries. Let's first solve over reals.
        
        # For a given R, this is underdetermined if 120R > 2304, i.e., R > 19.2
        # So R >= 20 should be feasible in principle, but the structure constraints
        # (sparsity, integrality) make it harder.
        
        # Let's try to find a rank-R decomposition by:
        # 1. Start with the rank-44 solution
        # 2. Try to merge pairs of terms into single terms
        
        # Actually, let's try a different strategy: use the known lower bounds
        # For 3x4 x 4x4, the rank is at least... let's think.
        
        # The rank of matrix multiplication M_{n,m,p} satisfies:
        # R >= max(n*m, m*p, p*n) = max(12, 16, 12) = 16
        # R >= n*p = 12 (from one factor)
        # Better bounds: R >= (n*m*p)/max(n,m,p) = 48/4 = 12
        # R >= n*m*p / (n+m+p-1) by some bounds...
        
        # Known results: R(M_{3,4,4}) is between 24 and 44 (trivial)
        # Actually, let's just try to find a good decomposition.
        
        # Let's use a randomized approach: try random rank-R decompositions
        # and check if they work
        
        for R in range(max_rank, min(best_rank - 1, 30), -1):
            if time.time() > deadline - 10:
                break
            
            # Try to find a rank-R decomposition
            # We'll use the fact that we can parameterize the solution
            
            # For each rank-1 term r, let u_r, v_r, w_r be the vectors
            # The constraint is: for all (i,j,k) with T[i,j,k] = 1:
            #   sum_r u_r[i] * v_r[j] * w_r[k] = 1
            # For all (i,j,k) with T[i,j,k] = 0:
            #   sum_r u_r[i] * v_r[j] * w_r[k] = 0
            
            # This is a nonlinear system. Let's try a greedy approach.
            
            # Greedy: at each step, find the "best" rank-1 term that reduces
            # the residual the most, then subtract it and repeat
            
            residual = T.copy()
            U_list = []
            V_list = []
            W_list = []
            
            for step in range(R):
                if time.time() > deadline - 5:
                    break
                
                # Find the rank-1 term that best approximates the residual
                # We'll search over a set of candidate terms
                
                best_term = None
                best_score = -np.inf
                
                # Candidate terms: based on the structure of T
                # For each (row_a, col_a, col_c), the natural term is:
                # u has 1 at row_a*4+col_a, v has 1 at col_a*4+col_c, w has 1 at col_c*3+row_a
                
                # But we can also have terms with multiple non-zero entries
                # Let's try terms with support size 1, 2, or 3 in each vector
                
                # For efficiency, let's try structured candidates
                candidates_local = []
                
                # Type 1: single-term (support size 1 in each)
                for row_a in range(3):
                    for col_a in range(4):
                        for col_c in range(4):
                            i = row_a * 4 + col_a
                            j = col_a * 4 + col_c
                            k = col_c * 3 + row_a
                            u = np.zeros(12); u[i] = 1
                            v = np.zeros(16); v[j] = 1
                            w = np.zeros(12); w[k] = 1
                            candidates_local.append((u, v, w))
                
                # Type 2: Strassen-like terms for 2x2 blocks
                # For each 2x2 block in A, B, C, add Strassen terms
                
                # Let's try to find a good term by checking how well it matches the residual
                # Score = sum over (i,j,k) of residual[i,j,k] * (u[i]*v[j]*w[k])
                
                for u, v, w in candidates_local:
                    term_tensor = np.einsum('i,j,k->ijk', u, v, w)
                    score = np.sum(residual * term_tensor)
                    # Also penalize terms that create large values where T=0
                    penalty = np.sum((1 - (T == 0)) * 0 + (T == 0) * term_tensor**2) * 0.1
                    total_score = score - penalty
                    if total_score > best_score:
                        best_score = total_score
                        best_term = (u, v, w)
                
                if best_term is None:
                    break
                
                u, v, w = best_term
                term_tensor = np.einsum('i,j,k->ijk', u, v, w)
                residual -= term_tensor
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
            
            if len(U_list) <= R:
                # Check if we've achieved a good approximation
                U_arr = np.array(U_list).T  # shape (12, len)
                V_arr = np.array(V_list).T  # shape (16, len)
                W_arr = np.array(W_list).T  # shape (12, len)
                
                # Verify
                computed = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
                error = np.max(np.abs(computed - T))
                
                if error < 1e-9:
                    # Found a valid decomposition!
                    # Clean up: remove any zero columns
                    nonzero_cols = np.any(U_arr != 0, axis=0) | np.any(V_arr != 0, axis=0) | np.any(W_arr != 0, axis=0)
                    U_arr = U_arr[:, nonzero_cols]
                    V_arr = V_arr[:, nonzero_cols]
                    W_arr = W_arr[:, nonzero_cols]
                    
                    final_rank = U_arr.shape[1]
                    if final_rank < best_rank:
                        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
        
        return None
    
    sol = get_current_solution()
    
    # Verify correctness
    np.random.seed(42)
    if not verify_solution(sol, max_tests=100):
        rank = 48
        U = np.zeros((12, rank))
        V = np.zeros((16, rank))
        W = np.zeros((12, rank))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    idx = (row_a * 4 + col_a) * 4 + col_c
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    U[i, idx] = 1
                    V[j, idx] = 1
                    W[k, idx] = 1
        sol = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better solution via sparse search
    if time.time() < deadline - 30:
        better_sol = try_sparse_search(best_rank=44, max_rank=40)
        if better_sol is not None:
            # Verify the better solution
            np.random.seed(42)
            if verify_solution(better_sol, max_tests=200):
                sol = better_sol
    
    # Check deadline
    if time.time() > deadline:
        return sol
    
    return sol
