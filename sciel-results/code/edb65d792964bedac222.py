import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    V_strassen = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, -1, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    C_strassen = [
        [[1, 0], [0, 1]], # P1
        [[0, 0], [1, -1]],# P2
        [[0, 1], [0, 1]], # P3
        [[1, 0], [1, 0]], # P4
        [[-1, 1], [0, 0]],# P5
        [[0, 0], [0, 1]], # P6
        [[1, 0], [0, 0]]  # P7
    ]
    
    col = 0
    # Blocks:
    # 1: A1 (rows 0-1, cols 0-1), B1 (rows 0-1, cols 0-1) -> C1 (rows 0-1, cols 0-1)
    # 2: A1, B2 (rows 0-1, cols 2-3) -> C2 (rows 0-1, cols 2-3)
    # 3: A2 (rows 0-1, cols 2-3), B3 (rows 2-3, cols 0-1) -> C3 (rows 0-1, cols 0-1)
    # 4: A2, B4 (rows 2-3, cols 2-3) -> C4 (rows 0-1, cols 2-3)
    
    blocks = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 2)
    ]
    
    for a_ro, a_co, b_ro, b_co, c_ro, c_co in blocks:
        for k in range(7):
            # Fill U
            # A indices: i*m + j
            # Local A indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
            # Global A indices: (a_ro + li, a_co + lj)
            # li, lj in {0,1}
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    val = U_strassen[k][local_idx]
                    if val != 0:
                        global_i = a_ro + li
                        global_j = a_co + lj
                        U[global_i * m + global_j][col] = val
            
            # Fill V
            # B indices: j*p + k
            # Local B indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
            # Global B indices: (b_ro + li, b_co + lj)
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    val = V_strassen[k][local_idx]
                    if val != 0:
                        global_j = b_ro + li
                        global_k = b_co + lj
                        V[global_j * p + global_k][col] = val
            
            # Fill W
            # W indices: k*n + i
            # C indices: (i, k)
            # Local C indices: (li, lj)
            # Global C indices: (c_ro + li, c_co + lj)
            for li in range(2):
                for lj in range(2):
                    val = C_strassen[k][li][lj]
                    if val != 0:
                        global_i = c_ro + li
                        global_k = c_co + lj
                        W[global_k * n + global_i][col] = val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
