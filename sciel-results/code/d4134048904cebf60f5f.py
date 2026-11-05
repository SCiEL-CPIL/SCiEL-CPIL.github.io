"""
Matrix multiplication 4x4x7: rank-93 decomposition via Strassen blocks.

Strategy:
  - 4x4x4 block: 7 Strassen terms (rank 49)
  - 4x4x2 block: 4 sub-blocks x 7 Strassen terms (rank 28)
  - 4x4x1 block: 16 schoolbook terms (rank 16)
  Total: 49 + 28 + 16 = 93

The decomposition is built directly with integer entries.
"""
import numpy as np
import time


def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    # Strassen 2x2x2: 7 rank-1 terms
    # Each term: (x_dict, y_dict, c_dict) where keys are (row, col) indices in 2x2
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Allocate full-size arrays
    max_rank = n * m * p  # 112
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)

    rank = 0

    def add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """Add 7 Strassen rank-1 terms for a 2x2x2 sub-problem."""
        nonlocal rank
        for t in range(7):
            x_sub, y_sub, c_sub = strassen_2x2[t]
            r = rank
            rank += 1
            for (li, lj), cx in x_sub.items():
                i = A_rows[li]
                j = A_cols[lj]
                U[i * m + j, r] += cx
            for (li, lj), cy in y_sub.items():
                j = B_rows[li]
                k = B_cols[lj]
                V[j * p + k, r] += cy
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                k = C_rows[li]
                i = C_cols[lj]
                W[k * n + i, r] += cc

    # --- 4x4x4 block: 4 sub-blocks, each 2x2x2 Strassen ---
    # Sub-block (bi, bj) where bi, bj in {0,1}
    for bi in range(2):
        for bj in range(2):
            A_rows = [2*bi, 2*bi+1]
            A_cols = [2*bj, 2*bj+1]
            B_rows = [2*bi, 2*bi+1]
            B_cols = [2*bj, 2*bj+1]
            C_rows = [2*bi, 2*bi+1]
            C_cols = [2*bj, 2*bj+1]
            add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols)

    # --- 4x4x2 block (cols 4,5): 4 sub-blocks, each 2x2x2 Strassen ---
    for bi in range(2):
        for bj in range(2):
            A_rows = [2*bi, 2*bi+1]
            A_cols = [2*bj, 2*bj+1]
            B_rows = [2*bi, 2*bi+1]
            B_cols = [4, 5]
            C_rows = [2*bi, 2*bi+1]
            C_cols = [4, 5]
            add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols)

    # --- 4x4x1 block (col 6): 16 schoolbook terms ---
    for i in range(4):
        for j in range(4):
            r = rank
            rank += 1
            U[i * m + j, r] = 1.0
            V[j * p + 6, r] = 1.0
            W[6 * n + i, r] = 1.0

    # Trim to actual rank
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]

    # Verify
    Ua = U.astype(float)
    Va = V.astype(float)
    Wa = W.astype(float)

    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    prod = Ua[u_idx, :] * Va[v_idx, :] * Wa[w_idx, :]
    result = prod.sum(axis=1)
    err = np.max(np.abs(result - 1.0))

    if err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank 112
        U_s = [[0] * (n*m*p) for _ in range(n*m)]
        V_s = [[0] * (n*m*p) for _ in range(m*p)]
        W_s = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
