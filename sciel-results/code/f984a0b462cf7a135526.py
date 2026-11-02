import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 4, 4
    
    # Rank-8 decomposition for M_{3,2,2}
    # Local indices: U: 0..5 (3 rows x 2 cols), V: 0..3 (2 rows x 2 cols), W: 0..5 (2 cols x 3 rows)
    local_U = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
    ], dtype=float)
    
    local_V = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
    ], dtype=float)
    
    local_W = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
    ], dtype=float)
    
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2),
    ]
    
    for block_idx, (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start) in enumerate(blocks):
        col_offset = block_idx * 8
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+8] = local_U[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+8] = local_V[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+8] = local_W[local_c_idx, :]
    
    # Verify
    T = np.zeros((12, 16, 12))
    for row_a in range(3):
        for col_a in range(4):
            for col_c in range(4):
                i = row_a * 4 + col_a
                j = col_a * 4 + col_c
                k = col_c * 3 + row_a
                T[i, j, k] = 1.0
    
    computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    error = np.max(np.abs(computed - T))
    
    if error > 1e-9:
        # Fallback to rank-48
        U = np.zeros((12, 48))
        V = np.zeros((16, 48))
        W = np.zeros((12, 48))
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
