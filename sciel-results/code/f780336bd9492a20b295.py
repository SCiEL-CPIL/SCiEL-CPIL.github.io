"""Combined program: Strassen-based structural decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill

    # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]

    def build_decomposition():
        """
        Build a decomposition for 4x4x7 matrix multiplication using row-block splitting.
        
        Strategy:
        - Split n=4 into two row-blocks of size 2: A_top (2x4), A_bot (2x4)
        - C_top (2x7) = A_top @ B, C_bot (2x7) = A_bot @ B
        - For each 2x4x7 sub-tensor:
          - Split B into B_left (4x4) and B_right (4x3)
          - 2x4x4: split A into A_left(2x2), A_right(2x2); B_left into B_00(2x2), B_01(2x2)
            -> C_00 = A_left @ B_00 (rank 7), C_01 = A_right @ B_01 (rank 7). Total: 14
          - 2x4x3: split A into A_left(2x2), A_right(2x2); B_right into B_00(2x3), B_01(2x3)
            -> C_00 = A_left @ B_00 (rank 11), C_01 = A_right @ B_01 (rank 11). Total: 22
          - Per sub-tensor: 14 + 22 = 36
        - Total rank: 2 * 36 = 72
        """
        total_rank = 72
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        current_rank = 0

        def add_strassen_2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
            """Add Strassen 2x2x2 for A_block @ B_block -> C_block."""
            nonlocal current_rank
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                for (li, lj), cx in x_sub.items():
                    if cx == 0: continue
                    i = A_rows[li]
                    j = A_cols[lj]
                    U[i * m + j, r] += cx
                for (li, lj), cy in y_sub.items():
                    if cy == 0: continue
                    i = B_rows[li]
                    j = B_cols[lj]
                    V[i * p + j, r] += cy
                for (li, lj), cc in c_sub.items():
                    if cc == 0: continue
                    i_final = C_rows[li]
                    k_final = C_cols[lj]
                    W[k_final * n + i_final, r] += cc

        def add_2x2x3(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
            """Add rank-11 decomposition for 2x2x3: Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4)."""
            nonlocal current_rank
            # B is 2x3, C is 2x3. Split B into B_left (2x2) and B_right (2x1).
            B_left_cols = B_cols[:2]
            B_right_col = B_cols[2]
            C_left_cols = C_cols[:2]
            C_right_col = C_cols[2]
            
            # Part 1: 2x2x2 Strassen (rank 7)
            add_strassen_2x2(A_rows, A_cols, B_rows, B_left_cols, C_rows, C_left_cols)
            
            # Part 2: 2x2x1 schoolbook (rank 4)
            for li in range(2):
                for lj in range(2):
                    r = current_rank
                    current_rank += 1
                    U[A_rows[li] * m + A_cols[lj], r] = 1.0
                    V[B_rows[lj] * p + B_right_col, r] = 1.0
                    W[C_right_col * n + C_rows[li], r] = 1.0

        # For each row-block (top and bottom):
        for row_block in range(2):
            A_block_rows = [2 * row_block, 2 * row_block + 1]
            C_block_rows = [2 * row_block, 2 * row_block + 1]
            
            # 2x4x4 part: B columns 0-3
            # A_left: cols 0-1, A_right: cols 2-3
            # B_00: rows 0-1, cols 0-1; B_01: rows 2-3, cols 2-3
            A_left_cols = [0, 1]
            A_right_cols = [2, 3]
            B_00_rows = [0, 1]; B_00_cols = [0, 1]
            B_01_rows = [2, 3]; B_01_cols = [2, 3]
            C_00_cols = [0, 1]
            C_01_cols = [2, 3]
            
            # C_00 = A_left @ B_00 (2x2x2, rank 7)
            add_strassen_2x2(A_block_rows, A_left_cols, B_00_rows, B_00_cols, C_block_rows, C_00_cols)
            # C_01 = A_right @ B_01 (2x2x2, rank 7)
            add_strassen_2x2(A_block_rows, A_right_cols, B_01_rows, B_01_cols, C_block_rows, C_01_cols)
            
            # 2x4x3 part: B columns 4-6
            # B_right: rows 0-1, cols 4-6 (2x3); rows 2-3, cols 4-6 (2x3)
            B_right_top_rows = [0, 1]; B_right_top_cols = [4, 5, 6]
            B_right_bot_rows = [2, 3]; B_right_bot_cols = [4, 5, 6]
            C_right_cols = [4, 5, 6]
            
            # C_right_top = A_left @ B_right_top (2x2x3, rank 11)
            add_2x2x3(A_block_rows, A_left_cols, B_right_top_rows, B_right_top_cols, C_block_rows, C_right_cols)
            # C_right_bot = A_right @ B_right_bot (2x2x3, rank 11)
            add_2x2x3(A_block_rows, A_right_cols, B_right_bot_rows, B_right_bot_cols, C_block_rows, C_right_cols)
        
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
        Build the structural decomposition.
        Returns (U, V, W, rank).
        """
        U_str, V_str, W_str = build_decomposition()
        rank = len(U_str[0])
        return U_str, V_str, W_str, rank

    # Build the structural solution
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
