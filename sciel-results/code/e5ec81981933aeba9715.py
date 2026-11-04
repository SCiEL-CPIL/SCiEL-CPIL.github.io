import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    # Rows of U_local correspond to a, b, c, d
    # Rows of V_local correspond to e, f, g, h
    # Rows of W_local correspond to C00, C01, C10, C11
    
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Define the 4 blocks
    # Each block is defined by:
    # A_j_offset: 0 or 2 (columns of A)
    # B_j_offset: 0 or 2 (rows of B)
    # B_k_offset: 0, 2, 4, 6 (columns of B)
    # C_k_offset: 0, 2, 4, 6 (columns of C)
    
    blocks = [
        # Block 1: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
        {'A_j': 0, 'B_j': 0, 'B_k': 0, 'C_k': 0},
        # Block 2: A[0:2, 2:4], B[2:4, 2:4], C[0:2, 2:4]
        {'A_j': 2, 'B_j': 2, 'B_k': 2, 'C_k': 2},
        # Block 3: A[0:2, 0:2], B[0:2, 4:6], C[0:2, 4:6]
        {'A_j': 0, 'B_j': 0, 'B_k': 4, 'C_k': 4},
        # Block 4: A[0:2, 2:4], B[2:4, 6:8], C[0:2, 6:8]
        {'A_j': 2, 'B_j': 2, 'B_k': 6, 'C_k': 6},
    ]
    
    col_offset = 0
    for block in blocks:
        aj = block['A_j']
        bj = block['B_j']
        bk = block['B_k']
        ck = block['C_k']
        
        # Global indices for A (i, j)
        # i=0,1; j=aj, aj+1
        # Flat index i*m + j
        a_indices = [
            0 * m + aj,       # a (0, aj)
            0 * m + (aj + 1), # b (0, aj+1)
            1 * m + aj,       # c (1, aj)
            1 * m + (aj + 1)  # d (1, aj+1)
        ]
        
        # Global indices for B (j, k)
        # j=bj, bj+1; k=bk, bk+1
        # Flat index j*p + k
        b_indices = [
            bj * p + bk,       # e (bj, bk)
            bj * p + (bk + 1), # f (bj, bk+1)
            (bj + 1) * p + bk,       # g (bj+1, bk)
            (bj + 1) * p + (bk + 1)  # h (bj+1, bk+1)
        ]
        
        # Global indices for C (i, k) -> Trace convention W[k*n + i, r]
        # i=0,1; k=ck, ck+1
        # Flat index k*n + i
        c_indices = [
            ck * n + 0,       # C00 (i=0, k=ck)
            (ck + 1) * n + 0, # C01 (i=0, k=ck+1)
            ck * n + 1,       # C10 (i=1, k=ck)
            (ck + 1) * n + 1  # C11 (i=1, k=ck+1)
        ]
        
        # Fill U
        for r in range(4):
            for c in range(7):
                if U_loc[r][c] != 0:
                    U[a_indices[r]][col_offset + c] += U_loc[r][c]
                    
        # Fill V
        for r in range(4):
            for c in range(7):
                if V_loc[r][c] != 0:
                    V[b_indices[r]][col_offset + c] += V_loc[r][c]
                    
        # Fill W
        for r in range(4):
            for c in range(7):
                if W_loc[r][c] != 0:
                    W[c_indices[r]][col_offset + c] += W_loc[r][c]
                    
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}
