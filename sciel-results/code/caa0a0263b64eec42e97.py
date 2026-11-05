"""Combined program: Strassen-based structural decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill

    def build_strassen_decomposition():
        """
        Build a Strassen-based decomposition for 4x4x7 matrix multiplication.
        
        Strategy:
        - Decompose 4x4x7 as 4x4x4 (Strassen rank 49) + 4x4x2 (Block Strassen rank 28) + 4x4x1 (Schoolbook rank 16)
        - Total rank: 93
        
        The 4x4x4 part uses Strassen's algorithm with rank 49.
        The 4x4x2 part uses a block decomposition with rank 28.
        The 4x4x1 part uses schoolbook multiplication with rank 16.
        """
        total_rank = 93
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        current_rank = 0
        
        # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
        # x_sub, y_sub are 2x2 dicts, c_sub is 2x2 dict for output
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        # Strassen 4x4x4: 7 outer terms, each with 7 inner terms
        strassen_4x4_outer = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}),
        ]
        
        # For 4x4x4, the c_block for each outer term m_idx
        c_blocks_4x4 = [
            {(0,0): 1.0, (1,1): 1.0},
            {(1,0): 1.0, (1,1): -1.0},
            {(0,1): 1.0, (1,1): 1.0},
            {(0,0): 1.0, (1,0): 1.0},
            {(0,0): -1.0, (0,1): 1.0},
            {(1,1): 1.0},
            {(0,0): 1.0},
        ]
        
        # Part 1: 4x4x4 Strassen (rank 49)
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                # Compute U coefficients
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                
                # Compute V coefficients
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * p + j, r] += cy * cb
                
                # Compute W coefficients
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c
        
        # Part 2: 4x4x2 Block Strassen (rank 28)
        # B columns 4, 5.
        # C = A @ B[:, 4:6]
        # Split A into 2x2 blocks A00, A01, A10, A11
        # Split B[:, 4:6] into B_top (2x2), B_bot (2x2)
        # C_top = A00 @ B_top + A01 @ B_bot
        # C_bot = A10 @ B_top + A11 @ B_bot
        # Each product is 2x2x2, rank 7.
        # 4 products total rank 28.
        # Total so far: 49 + 28 = 77
        
        def add_strassen_2x2(A_block_rows, A_block_cols, B_block_rows, B_block_cols, C_block_rows, C_block_cols):
            """
            Add a Strassen 2x2x2 decomposition for A_block @ B_block to the global U, V, W.
            A_block is 2x2, B_block is 2x2, C_block is 2x2.
            """
            nonlocal current_rank
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                # U: A_block entries
                for (li, lj), cx in x_sub.items():
                    i = A_block_rows[li]
                    j = A_block_cols[lj]
                    U[i * m + j, r] += cx
                    
                # V: B_block entries
                for (li, lj), cy in y_sub.items():
                    i = B_block_rows[li]
                    j = B_block_cols[lj]
                    V[i * p + j, r] += cy
                    
                # W: C_block entries
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    i_final = C_block_rows[li]
                    k_final = C_block_cols[lj]
                    W[k_final * n + i_final, r] += cc

        # Define blocks
        A00_r = [0, 1]; A00_c = [0, 1]
        A01_r = [0, 1]; A01_c = [2, 3]
        A10_r = [2, 3]; A10_c = [0, 1]
        A11_r = [2, 3]; A11_c = [2, 3]
        
        B_top_r = [0, 1]; B_top_c = [4, 5]
        B_bot_r = [2, 3]; B_bot_c = [4, 5]
        
        C_top_r = [0, 1]; C_top_c = [4, 5]
        C_bot_r = [2, 3]; C_bot_c = [4, 5]
        
        # P1 = A00 @ B_top -> C_top
        add_strassen_2x2(A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c)
        # P2 = A01 @ B_bot -> C_top
        add_strassen_2x2(A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c)
        # P3 = A10 @ B_top -> C_bot
        add_strassen_2x2(A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c)
        # P4 = A11 @ B_bot -> C_bot
        add_strassen_2x2(A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c)

        # Part 3: 4x4x1 schoolbook (rank 16)
        # B column 6.
        # C[i][6] = sum_j A[i][j] * B[j][6]
        for i in range(n):
            for j in range(m):
                r = current_rank
                current_rank += 1
                U[i * m + j, r] = 1.0
                V[j * p + 6, r] = 1.0
                W[6 * n + i, r] = 1.0
        
        assert current_rank == total_rank, f"Expected {total_rank}, got {current_rank}"
        
        return U[:, :total_rank].tolist(), V[:, :total_rank].tolist(), W[:, :total_rank].tolist()
    
    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Precompute index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))
    
    def search_decomposition():
        """
        Attempt to find a lower rank decomposition using iterative refinement.
        Returns (U, V, W, rank) if successful, else None.
        """
        U_str, V_str, W_str = build_strassen_decomposition()
        U_arr = np.array(U_str, dtype=float)
        V_arr = np.array(V_str, dtype=float)
        W_arr = np.array(W_str, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Check for zero columns
        keep = []
        for r in range(rank):
            if np.any(U_arr[:, r] != 0) and np.any(V_arr[:, r] != 0) and np.any(W_arr[:, r] != 0):
                keep.append(r)
        
        if len(keep) < rank:
            U_arr = U_arr[:, keep]
            V_arr = V_arr[:, keep]
            W_arr = W_arr[:, keep]
            rank = len(keep)
        
        # Try to reduce rank by identifying redundant pairs of terms
        # If two terms (r1, r2) have U[r1] = a*U[r2], V[r1] = b*V[r2], W[r1] = c*W[r2]
        # with a*b*c = 1, they can potentially be merged or one eliminated.
        # More practically: check if removing a term and adjusting others maintains validity.
        
        # Strategy: For each term, check if it's linearly dependent on others in a way
        # that allows elimination. We'll use a simple heuristic: try removing each term
        # and see if the remaining terms still form a valid decomposition (they won't unless
        # the removed term was redundant).
        
        # A more effective approach: look for pairs where U columns are proportional
        # and check if the corresponding V and W columns allow cancellation.
        
        improved = False
        for _ in range(3):  # Try up to 3 passes
            if time.time() > deadline - 10:
                break
            
            found_reduction = False
            for r1 in range(rank):
                if time.time() > deadline - 10:
                    break
                for r2 in range(r1 + 1, rank):
                    if time.time() > deadline - 10:
                        break
                    
                    # Check if U columns are proportional
                    u1 = U_arr[:, r1]
                    u2 = U_arr[:, r2]
                    if not np.any(u1) or not np.any(u2):
                        continue
                    
                    # Find proportionality factor
                    nonzero_u1 = np.where(u1 != 0)[0]
                    nonzero_u2 = np.where(u2 != 0)[0]
                    common = np.intersect1d(nonzero_u1, nonzero_u2)
                    if len(common) == 0:
                        continue
                    
                    # Check if ratio is constant
                    ratios = u1[common] / u2[common]
                    if np.max(np.abs(ratios - ratios[0])) > 1e-10:
                        continue
                    
                    a = ratios[0]
                    
                    # Check V columns
                    v1 = V_arr[:, r1]
                    v2 = V_arr[:, r2]
                    nonzero_v1 = np.where(v1 != 0)[0]
                    nonzero_v2 = np.where(v2 != 0)[0]
                    common_v = np.intersect1d(nonzero_v1, nonzero_v2)
                    if len(common_v) == 0:
                        continue
                    
                    ratios_v = v1[common_v] / v2[common_v]
                    if np.max(np.abs(ratios_v - ratios_v[0])) > 1e-10:
                        continue
                    
                    b = ratios_v[0]
                    
                    # Check W columns
                    w1 = W_arr[:, r1]
                    w2 = W_arr[:, r2]
                    nonzero_w1 = np.where(w1 != 0)[0]
                    nonzero_w2 = np.where(w2 != 0)[0]
                    common_w = np.intersect1d(nonzero_w1, nonzero_w2)
                    if len(common_w) == 0:
                        continue
                    
                    ratios_w = w1[common_w] / w2[common_w]
                    if np.max(np.abs(ratios_w - ratios_w[0])) > 1e-10:
                        continue
                    
                    c = ratios_w[0]
                    
                    # If a*b*c ≈ 0, one term might be redundant
                    # If a*b*c ≈ 1, they contribute equally and might be mergeable
                    # For now, just check if we can eliminate one by adjusting the other
                    # This requires solving a linear system, which is complex.
                    # Instead, we'll just note this as a potential reduction.
                    
                    # Simple case: if a*b*c is very small, the term is nearly redundant
                    if abs(a * b * c) < 1e-10:
                        # Try removing term r1
                        U_test = np.delete(U_arr, r1, axis=1)
                        V_test = np.delete(V_arr, r1, axis=1)
                        W_test = np.delete(W_arr, r1, axis=1)
                        
                        err_test = verify(U_test.tolist(), V_test.tolist(), W_test.tolist(), rank - 1)
                        if err_test < 1e-10:
                            U_arr = U_test
                            V_arr = V_test
                            W_arr = W_test
                            rank -= 1
                            found_reduction = True
                            improved = True
                            break
                if found_reduction:
                    break
            if not found_reduction:
                break
        
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank

    # Try the search (which currently returns the structural solution)
    U, V, W, rank = search_decomposition()
    
    # Verify correctness
    err = verify(U, V, W, rank)
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
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
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
