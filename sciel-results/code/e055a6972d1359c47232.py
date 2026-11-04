import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = 4, 4, 8
    rank_target = 98

    U = np.zeros((n * m, rank_target), dtype=np.int8)
    V = np.zeros((m * p, rank_target), dtype=np.int8)
    W = np.zeros((p * n, rank_target), dtype=np.int8)

    # Strassen 2x2x2 decomposition (7 terms)
    U_2 = np.zeros((4, 7), dtype=np.int8)
    V_2 = np.zeros((4, 7), dtype=np.int8)
    W_2 = np.zeros((4, 7), dtype=np.int8)

    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]

    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]

    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]

    col = 0
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            val = coeff_a * int(U_2[a_eff_idx, m_s])
                            if val != 0:
                                U[global_a_idx, col] += val

                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            val = coeff_b * int(V_2[b_eff_idx, m_s])
                            if val != 0:
                                V[global_b_idx, col] += val

                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            val = coeff_c * int(W_2[p_eff_idx, m_s])
                            if val != 0:
                                W[global_w_idx, col] += val

                col += 1

    # Now try to reduce rank by finding linear dependencies among the 98 columns
    # The key insight from the inspiration: U_left == U_right, so we have structure
    # to exploit. Let's check if we can merge columns.
    
    if time.time() < deadline - 10:
        # Try to find if any columns are linearly dependent using rank reduction
        U_f = U.astype(np.float64)
        V_f = V.astype(np.float64)
        W_f = W.astype(np.float64)
        
        # The tensor T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # If we can find a linear combination of columns that gives zero contribution,
        # we can eliminate that column.
        
        # Check if the 98 columns of U have rank < 98
        # U is 16 x 98, so max rank is 16. There must be many dependencies.
        # But we need dependencies that also work for V and W simultaneously.
        
        # A column r can be eliminated if there exist coefficients c_s (s != r) 
        # such that sum_s c_s * U[:,s] = U[:,r], sum_s c_s * V[:,s] = V[:,r], 
        # sum_s c_s * W[:,s] = W[:,r]
        
        # This is equivalent to: the vector (U[:,r], V[:,r], W[:,r]) is in the span
        # of {(U[:,s], V[:,s], W[:,s]) : s != r}
        
        # Build the combined matrix and check rank
        # Combined is (16+32+32) x 98 = 80 x 98
        combined = np.vstack([U_f, V_f, W_f])  # 80 x 98
        
        # Compute rank of combined
        rank_combined = np.linalg.matrix_rank(combined)
        
        if rank_combined < 98:
            # There are dependencies. Find them.
            # Use SVD to find null space
            U_svd, S, Vt = np.linalg.svd(combined, full_matrices=False)
            # Null space is in the last (98 - rank_combined) rows of Vt
            null_dim = 98 - rank_combined
            null_space = Vt[rank_combined:, :].T  # 98 x null_dim
            
            # For each null vector, find if it has a single large component
            # that we can use to eliminate a column
            # Actually, we need to be more careful. Let's try a different approach.
            
            # Try to find pairs of columns that are close to linear combinations
            # of each other plus other columns.
            
            # Simpler approach: try to find if any column is a linear combination
            # of the first k columns for some k < 98
            # This is expensive, so let's try a heuristic.
            
            # Check if columns come in pairs that differ only in V or W
            # Given U_left == U_right, columns 0-48 and 49-97 have same U.
            # So for column r and column r+49, U[:,r] == U[:,r+49]
            # If V[:,r] + V[:,r+49] and W[:,r] + W[:,r+49] could be combined...
            
            # Actually, the structure is:
            # For r in 0..48: U[:,r] = U_left[:,r], V[:,r] = V_left[:,r], W[:,r] = W_left[:,r]
            # For r in 49..97: U[:,r] = U_right[:,r] = U_left[:,r-49], V[:,r] = V_right[:,r-49], W[:,r] = W_right[:,r-49]
            
            # The tensor contribution of column r (left) and column r+49 (right) is:
            # U_left[:,r] ⊗ V_left[:,r] ⊗ W_left[:,r] + U_left[:,r] ⊗ V_right[:,r] ⊗ W_right[:,r]
            # = U_left[:,r] ⊗ (V_left[:,r] ⊗ W_left[:,r] + V_right[:,r] ⊗ W_right[:,r])
            
            # This is a sum of two rank-1 tensors sharing the same U factor.
            # We cannot generally combine them into a single rank-1 tensor.
            
            # However, if V_left[:,r] = V_right[:,r] and W_left[:,r] = W_right[:,r],
            # then we can combine: 2 * U_left[:,r] ⊗ V_left[:,r] ⊗ W_left[:,r]
            # But in general they are different.
            
            # Let's check if there's a simpler reduction.
            # The known minimum rank for 4x4x4 is 49 (Strassen).
            # For 4x4x8, the lower bound is max(49, 4*8) = 49? No, that's not right.
            # The rank of A(4,4,8) >= rank of A(4,4,4) = 49.
            # Also rank >= max(n*m, m*p, p*n) / something... 
            # Actually rank >= max(n*m, m*p, p*n) is not a lower bound.
            # The trivial lower bound is max(n*m, m*p, p*n) = 32.
            # But we know rank(4,4,4) = 49, so rank(4,4,8) >= 49.
            
            # Can we do better than 98? The known best for 4x4x8 is... 
            # Let me just return the rank-98 solution since I'm not confident in a reduction.
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
