"""Combined program: Strassen-based structural decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p

    def build_strassen_decomposition():
        """
        Build a Strassen-based decomposition for 4x4x7 matrix multiplication.
        
        Strategy:
        - Decompose 4x4x7 as 4x4x4 (Strassen rank 49) + 4x4x2 (Block Strassen rank 28) + 4x4x1 (Schoolbook rank 16)
        - Total rank: 93
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
        
        assert current_rank == total_rank, f"Expected rank {total_rank}, got {current_rank}"
        
        return U.tolist(), V.tolist(), W.tolist(), total_rank

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Precompute index arrays for efficient verification
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

    # Try the structural decomposition
    try:
        U_struct, V_struct, W_struct, rank_struct = build_strassen_decomposition()
        err = verify(U_struct, V_struct, W_struct, rank_struct)
        if err < 1e-8:
            best_rank = rank_struct
            best_result = {
                "U": U_struct,
                "V": V_struct,
                "W": W_struct
            }
    except Exception as e:
        pass

    # If we have a good result, try to refine it using a gradient-free search
    # that respects the structural constraints
    if best_rank < n * m * p and time.time() < deadline - 20:
        Ua = np.array(best_result["U"], dtype=float)
        Va = np.array(best_result["V"], dtype=float)
        Wa = np.array(best_result["W"], dtype=float)
        r = best_rank
        
        # Precompute index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        def compute_error(U_mat, V_mat, W_mat):
            """Compute max error of tensor decomposition."""
            u_vals = U_mat[u_idx, :]
            v_vals = V_mat[v_idx, :]
            w_vals = W_mat[w_idx, :]
            prod = u_vals * v_vals * w_vals
            result = prod.sum(axis=1)
            return np.max(np.abs(result - 1.0))
        
        def try_zero_column(U_mat, V_mat, W_mat, col_idx):
            """
            Try to zero out column col_idx by redistributing its contribution
            to other columns. Returns (success, new_U, new_V, new_W) or None.
            """
            if time.time() > deadline - 5:
                return None
            
            U_c = U_mat[:, col_idx].copy()
            V_c = V_mat[:, col_idx].copy()
            W_c = W_mat[:, col_idx].copy()
            
            # Check if column is already zero
            if np.all(U_c == 0) and np.all(V_c == 0) and np.all(W_c == 0):
                return (True, U_mat, V_mat, W_mat)
            
            # Try to find a combination of other columns that can absorb this one
            # For each other column, check if we can add a multiple to cancel
            for other_col in range(r):
                if other_col == col_idx:
                    continue
                
                U_o = U_mat[:, other_col]
                V_o = V_mat[:, other_col]
                W_o = W_mat[:, other_col]
                
                # Skip if other column is zero
                if np.all(U_o == 0) and np.all(V_o == 0) and np.all(W_o == 0):
                    continue
                
                # Try to find alpha such that U_c + alpha*U_o ≈ 0, etc.
                # This requires U_c ≈ -alpha*U_o, V_c ≈ -alpha*V_o, W_c ≈ -alpha*W_o
                # Find alpha from U if possible
                alpha_candidates = set()
                
                # From U
                for i in range(len(U_c)):
                    if U_c[i] != 0 and U_o[i] != 0:
                        alpha_candidates.add(-U_c[i] / U_o[i])
                
                # From V
                for i in range(len(V_c)):
                    if V_c[i] != 0 and V_o[i] != 0:
                        alpha_candidates.add(-V_c[i] / V_o[i])
                
                # From W
                for i in range(len(W_c)):
                    if W_c[i] != 0 and W_o[i] != 0:
                        alpha_candidates.add(-W_c[i] / W_o[i])
                
                # Check each candidate alpha
                for alpha in alpha_candidates:
                    # Round alpha to nearest 0.5
                    alpha_rounded = np.round(alpha * 2) / 2
                    
                    # Check if this alpha works for all three matrices
                    U_new_c = U_c + alpha_rounded * U_o
                    V_new_c = V_c + alpha_rounded * V_o
                    W_new_c = W_c + alpha_rounded * W_o
                    
                    # Check if close to zero
                    if (np.max(np.abs(U_new_c)) < 1e-8 and 
                        np.max(np.abs(V_new_c)) < 1e-8 and 
                        np.max(np.abs(W_new_c)) < 1e-8):
                        # Success! Zero out the column
                        U_new = U_mat.copy()
                        V_new = V_mat.copy()
                        W_new = W_mat.copy()
                        U_new[:, col_idx] = 0
                        V_new[:, col_idx] = 0
                        W_new[:, col_idx] = 0
                        return (True, U_new, V_new, W_new)
            
            return None
        
        # Iteratively try to eliminate columns
        eliminated = True
        while eliminated and time.time() < deadline - 10:
            eliminated = False
            for col in range(r):
                if time.time() > deadline - 5:
                    break
                
                result = try_zero_column(Ua, Va, Wa, col)
                if result is not None:
                    success, Ua, Va, Wa = result
                    # Check if column is now zero
                    if np.all(Ua[:, col] == 0) and np.all(Va[:, col] == 0) and np.all(Wa[:, col] == 0):
                        # Remove this column
                        mask = np.ones(r, dtype=bool)
                        mask[col] = False
                        Ua = Ua[:, mask]
                        Va = Va[:, mask]
                        Wa = Wa[:, mask]
                        r -= 1
                        eliminated = True
                        break
            
            # Also try pairwise column combination: for each pair (i,j),
            # check if U[:,i] + s*U[:,j], V[:,i] + s*V[:,j], W[:,i] + s*W[:,j]
            # is zero for some s in {-2,-1.5,-1,-0.5,0,0.5,1,1.5,2}
            if not eliminated and time.time() < deadline - 10:
                for i in range(r):
                    if time.time() > deadline - 5:
                        break
                    if np.all(Ua[:, i] == 0) and np.all(Va[:, i] == 0) and np.all(Wa[:, i] == 0):
                        continue
                    for j in range(r):
                        if j == i:
                            continue
                        if np.all(Ua[:, j] == 0) and np.all(Va[:, j] == 0) and np.all(Wa[:, j] == 0):
                            continue
                        for s in [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]:
                            U_comb = Ua[:, i] + s * Ua[:, j]
                            V_comb = Va[:, i] + s * Va[:, j]
                            W_comb = Wa[:, i] + s * Wa[:, j]
                            if (np.max(np.abs(U_comb)) < 1e-8 and 
                                np.max(np.abs(V_comb)) < 1e-8 and 
                                np.max(np.abs(W_comb)) < 1e-8):
                                # Column i can be eliminated by adding s*column j
                                Ua[:, i] = 0
                                Va[:, i] = 0
                                Wa[:, i] = 0
                                mask = np.ones(r, dtype=bool)
                                mask[i] = False
                                Ua = Ua[:, mask]
                                Va = Va[:, mask]
                                Wa = Wa[:, mask]
                                r -= 1
                                eliminated = True
                                break
                        if eliminated:
                            break
                    if eliminated:
                        break
        
        # Final verification
        if r < best_rank:
            err = compute_error(Ua, Va, Wa)
            if err < 1e-8:
                best_rank = r
                best_result = {
                    "U": Ua.tolist(),
                    "V": Va.tolist(),
                    "W": Wa.tolist()
                }
        
        # Also try local perturbations to see if we can find a lower-rank solution
        if time.time() < deadline - 15:
            np.random.seed(42)
            # Try systematic small perturbations on individual entries
            for trial in range(60):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random column and entry to perturb
                col = np.random.randint(0, r)
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    row = np.random.randint(0, n * m)
                    U_new = Ua.copy()
                    delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    U_new[row, col] += delta
                    U_new = np.round(U_new * 2) / 2
                    err = compute_error(U_new, Va, Wa)
                elif mat_choice == 1:
                    row = np.random.randint(0, m * p)
                    V_new = Va.copy()
                    delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    V_new[row, col] += delta
                    V_new = np.round(V_new * 2) / 2
                    err = compute_error(Ua, V_new, Wa)
                else:
                    row = np.random.randint(0, p * n)
                    W_new = Wa.copy()
                    delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    W_new[row, col] += delta
                    W_new = np.round(W_new * 2) / 2
                    err = compute_error(Ua, Va, W_new)
                
                if err < 1e-8:
                    # Check if this creates a zero column
                    if mat_choice == 0:
                        if np.all(U_new[:, col] == 0):
                            mask = np.ones(r, dtype=bool)
                            mask[col] = False
                            U_red = U_new[:, mask]
                            V_red = Va[:, mask]
                            W_red = Wa[:, mask]
                            r_new = np.sum(mask)
                            if r_new < best_rank:
                                best_rank = r_new
                                best_result = {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
                    elif mat_choice == 1:
                        if np.all(V_new[:, col] == 0):
                            mask = np.ones(r, dtype=bool)
                            mask[col] = False
                            U_red = Ua[:, mask]
                            V_red = V_new[:, mask]
                            W_red = Wa[:, mask]
                            r_new = np.sum(mask)
                            if r_new < best_rank:
                                best_rank = r_new
                                best_result = {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
                    else:
                        if np.all(W_new[:, col] == 0):
                            mask = np.ones(r, dtype=bool)
                            mask[col] = False
                            U_red = Ua[:, mask]
                            V_red = Va[:, mask]
                            W_red = W_new[:, mask]
                            r_new = np.sum(mask)
                            if r_new < best_rank:
                                best_rank = r_new
                                best_result = {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
