"""
Rank-minimal tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Start with Strassen's rank-7 algorithm on 2x2 blocks (rank 42),
then attempt to reduce rank by finding linear dependencies or using
a direct search for lower-rank decompositions.
"""

import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # ---------------------------------------------------------------
    # Strassen 2x2@2x2 rank-7 decomposition
    # ---------------------------------------------------------------
    strassen_U = [
        {0: 1, 3: 1},       # P1: a00 + a11
        {2: 1, 3: 1},       # P2: a10 + a11
        {0: 1},             # P3: a00
        {3: 1},             # P4: a11
        {0: 1, 1: 1},       # P5: a00 + a01
        {0: -1, 2: 1},      # P6: a10 - a00
        {1: 1, 3: -1},      # P7: a01 - a11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # b00 + b11
        {0: 1},             # b00
        {1: 1, 3: -1},      # b01 - b11
        {2: 1, 0: -1},      # b10 - b00
        {3: 1},             # b11
        {0: 1, 1: 1},       # b00 + b01
        {2: 1, 3: 1},       # b10 + b11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c00 + c11
        {2: 1, 3: -1},      # P2: c10 - c11
        {1: 1, 3: 1},       # P3: c01 + c11
        {0: 1, 2: 1},       # P4: c00 + c10
        {0: -1, 1: 1},      # P5: -c00 + c01
        {3: 1},             # P6: c11
        {0: 1},             # P7: c00
    ]

    # ---------------------------------------------------------------
    # Build the rank-42 decomposition
    # ---------------------------------------------------------------
    rank_total = 42
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0
                v_row_offset = 0
            else:
                u_col_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                for local_idx, sign in strassen_U[prod_idx].items():
                    i_local = local_idx // 2
                    j_local = local_idx % 2
                    u_global = i_local * m + (u_col_offset + j_local)
                    U[u_global][col] += sign

                for local_idx, sign in strassen_V[prod_idx].items():
                    j_local = local_idx // 2
                    k_local = local_idx % 2
                    v_global = (v_row_offset + j_local) * p + (2 * j_block + k_local)
                    V[v_global][col] += sign

                for local_idx, sign in strassen_W[prod_idx].items():
                    i_local = local_idx // 2
                    k_local = local_idx % 2
                    w_global = (2 * j_block + k_local) * n + i_local
                    W[w_global][col] += sign

            col_offset += 7

    # ---------------------------------------------------------------
    # Verify the rank-42 decomposition
    # ---------------------------------------------------------------
    def verify(U_mat, V_mat, W_mat, rank):
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U_arr[idx_u] * V_arr[idx_v] * W_arr[idx_w])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err

    err_42 = verify(U, V, W, rank_total)
    if err_42 > 1e-9:
        # Fallback: schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(nm)]
        V_s = [[0] * rank_school for _ in range(mp)]
        W_s = [[0] * rank_school for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    best_U, best_V, best_W = U, V, W
    best_rank = rank_total

    # ---------------------------------------------------------------
    # Try to reduce rank by checking for linear dependencies
    # ---------------------------------------------------------------
    # For each pair of columns, check if we can merge them
    # by finding alpha, beta such that the combined column
    # maintains the tensor equation.
    
    if time.time() < deadline - 30:
        # Convert to numpy for efficient operations
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each pair of columns (c1, c2), check if there exist
        # alpha, beta (not both zero) such that the combination
        # alpha*col_c1 + beta*col_c2 is linearly dependent with
        # the remaining columns in a way that preserves the tensor.
        
        # Simpler approach: check if any column is a linear combination
        # of others. If so, we can eliminate it.
        
        # Check rank of U, V, W matrices
        rank_U = np.linalg.matrix_rank(U_arr)
        rank_V = np.linalg.matrix_rank(V_arr)
        rank_W = np.linalg.matrix_rank(W_arr)
        
        # If any matrix has rank < 42, we might be able to reduce
        # the overall rank.
        
        # Try greedy elimination: for each column, check if it's
        # in the span of the others.
        for col_to_remove in range(rank_total - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            # Check if column col_to_remove is in the span of others
            U_others = np.delete(U_arr, col_to_remove, axis=1)
            V_others = np.delete(V_arr, col_to_remove, axis=1)
            W_others = np.delete(W_arr, col_to_remove, axis=1)
            
            U_col = U_arr[:, col_to_remove]
            V_col = V_arr[:, col_to_remove]
            W_col = W_arr[:, col_to_remove]
            
            # Check if U_col is in span of U_others
            if np.linalg.matrix_rank(np.column_stack([U_others, U_col])) == np.linalg.matrix_rank(U_others):
                # U_col is in span, check V and W
                if np.linalg.matrix_rank(np.column_stack([V_others, V_col])) == np.linalg.matrix_rank(V_others):
                    if np.linalg.matrix_rank(np.column_stack([W_others, W_col])) == np.linalg.matrix_rank(W_others):
                        # All three columns are in span, can remove
                        U_new = np.delete(U_arr, col_to_remove, axis=1)
                        V_new = np.delete(V_arr, col_to_remove, axis=1)
                        W_new = np.delete(W_arr, col_to_remove, axis=1)
                        
                        new_rank = rank_total - 1
                        # Verify
                        err = verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), new_rank)
                        if err < 1e-9:
                            best_rank = new_rank
                            best_U = U_new.tolist()
                            best_V = V_new.tolist()
                            best_W = W_new.tolist()
                            U_arr = U_new
                            V_arr = V_new
                            W_arr = W_new
                            rank_total = new_rank
                            # Continue trying to remove more columns
                            col_to_remove = min(col_to_remove, rank_total - 1)
    
    # ---------------------------------------------------------------
    # Return the best decomposition found
    # ---------------------------------------------------------------
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
