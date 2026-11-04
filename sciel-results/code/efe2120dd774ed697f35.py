"""Optimized Strassen-based decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100

    # Pre-defined Strassen 2x2x2 coefficients
    # Format: (U_coeffs, V_coeffs, W_coeffs)
    # Indices are (row, col) for 2x2
    S = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]

    # Strassen 4x4x4 outer structure
    # x_def, y_def map block indices (0:00, 1:01, 2:10, 3:11) to coefficients
    # c_block maps output block indices to coefficients
    OUTER = [
        ({0:1, 3:1}, {0:1, 3:1}, {(0,0):1, (1,1):1}),
        ({2:1, 3:1}, {0:1}, {(1,0):1, (1,1):-1}),
        ({0:1}, {1:1, 3:-1}, {(0,1):1, (1,1):1}),
        ({3:1}, {2:1, 0:-1}, {(0,0):1, (1,0):1}),
        ({0:1, 1:1}, {3:1}, {(0,0):-1, (0,1):1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {(1,1):1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {(0,0):1})
    ]

    total_rank = 93
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0

    # Helper to add a 2x2x2 Strassen block
    def add_2x2_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        for s in S:
            u_c, v_c, w_c = s
            r = col
            col += 1
            for (li, lj), val in u_c.items():
                i = a_rows[li]
                j = a_cols[lj]
                U[i * m + j, r] += val
            for (li, lj), val in v_c.items():
                i = b_rows[li]
                j = b_cols[lj]
                V[i * p + j, r] += val
            for (li, lj), val in w_c.items():
                i = c_rows[li]
                k = c_cols[lj]
                W[k * n + i, r] += val

    # Part 1: 4x4x4 (Rank 49)
    # Iterate over 7 outer terms
    for o_idx in range(7):
        x_def, y_def, c_block = OUTER[o_idx]
        for s_idx in range(7):
            u_c, v_c, w_c = S[s_idx]
            r = col
            col += 1
            
            # U coefficients
            for (li, lj), cx in u_c.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb

            # V coefficients
            for (li, lj), cy in v_c.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb

            # W coefficients
            for (li, lj), cc in w_c.items():
                if cc == 0: continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c

    # Part 2: 4x4x2 (Rank 28)
    # B columns 4, 5
    # C rows 0,1 (top) and 2,3 (bot)
    # A blocks
    A00_r = [0, 1]; A00_c = [0, 1]
    A01_r = [0, 1]; A01_c = [2, 3]
    A10_r = [2, 3]; A10_c = [0, 1]
    A11_r = [2, 3]; A11_c = [2, 3]
    
    B_top_r = [0, 1]; B_top_c = [4, 5]
    B_bot_r = [2, 3]; B_bot_c = [4, 5]
    
    C_top_r = [0, 1]; C_top_c = [4, 5]
    C_bot_r = [2, 3]; C_bot_c = [4, 5]

    add_2x2_strassen(A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c)
    add_2x2_strassen(A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c)
    add_2x2_strassen(A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c)
    add_2x2_strassen(A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c)

    # Part 3: 4x4x1 (Rank 16)
    # B column 6
    for i in range(n):
        for j in range(m):
            r = col
            col += 1
            U[i * m + j, r] = 1.0
            V[j * p + 6, r] = 1.0
            W[6 * n + i, r] = 1.0

    assert col == total_rank, f"Rank mismatch: {col} != {total_rank}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Quick verification
    Ua = np.array(U_list)
    Va = np.array(V_list)
    Wa = np.array(W_list)
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    err = np.max(np.abs((Ua[u_idx] * Va[v_idx] * Wa[w_idx]).sum(axis=1) - 1.0))
    
    if err > 1e-8:
        # Fallback to schoolbook if something went wrong
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][c] = 1
                    V_s[j * p + k][c] = 1
                    W_s[k * n + i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
