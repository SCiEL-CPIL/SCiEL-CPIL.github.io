import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen local vectors for 2x2
    # U_local: 4 entries
    # V_local: 4 entries
    # W_local: 4 entries
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Sub-problem definitions
    # (A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start)
    # Note: A is 2x4, B is 4x4, C is 4x2 (p x n)
    # C index is k*n + i. k is col of C (0..3), i is row of C (0..1).
    
    subs = [
        (0, 0, 0, 0, 0, 0), # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 2, 0, 2), # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 2, 2, 0, 0, 0), # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 2, 0, 2)  # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    ]
    
    col_offset = 0
    for s in subs:
        a_r, a_c, b_r, b_c, c_r, c_c = s
        
        # Mappings
        # Local A indices 0,1,2,3 correspond to:
        # 0: (a_r, a_c), 1: (a_r, a_c+1), 2: (a_r+1, a_c), 3: (a_r+1, a_c+1)
        a_idx_map = [
            a_r * m + a_c,
            a_r * m + (a_c + 1),
            (a_r + 1) * m + a_c,
            (a_r + 1) * m + (a_c + 1)
        ]
        
        # Local B indices 0,1,2,3 correspond to:
        # 0: (b_r, b_c), 1: (b_r, b_c+1), 2: (b_r+1, b_c), 3: (b_r+1, b_c+1)
        b_idx_map = [
            b_r * p + b_c,
            b_r * p + (b_c + 1),
            (b_r + 1) * p + b_c,
            (b_r + 1) * p + (b_c + 1)
        ]
        
        # Local C indices 0,1,2,3 correspond to:
        # 0: (c_c, c_r), 1: (c_c+1, c_r), 2: (c_c, c_r+1), 3: (c_c+1, c_r+1)
        # C index is k*n + i. k is col, i is row.
        c_idx_map = [
            c_c * n + c_r,
            (c_c + 1) * n + c_r,
            c_c * n + (c_r + 1),
            (c_c + 1) * n + (c_r + 1)
        ]
        
        for k in range(7):
            col = col_offset + k
            # Fill U
            for local_i, val in enumerate(strassen_U[k]):
                if val != 0:
                    U[a_idx_map[local_i]][col] = val
            
            # Fill V
            for local_i, val in enumerate(strassen_V[k]):
                if val != 0:
                    V[b_idx_map[local_i]][col] = val
            
            # Fill W
            for local_i, val in enumerate(strassen_W[k]):
                if val != 0:
                    W[c_idx_map[local_i]][col] = val
                    
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}
