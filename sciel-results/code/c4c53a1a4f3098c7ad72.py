import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    
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
    
    # Iterate over blocks
    # J: 0..1 (blocks of j)
    # K: 0..3 (blocks of k)
    for J in range(m // 2):
        for K in range(p // 2):
            col_offset = (J * (p // 2) + K) * 7
            
            # Global indices for A (i, j)
            # i=0,1; j=2J, 2J+1
            # Flat index i*m + j
            a_indices = [
                0 * m + 2 * J,     # a (0, 2J)
                0 * m + (2 * J + 1), # b (0, 2J+1)
                1 * m + 2 * J,     # c (1, 2J)
                1 * m + (2 * J + 1)  # d (1, 2J+1)
            ]
            
            # Global indices for B (j, k)
            # j=2J, 2J+1; k=2K, 2K+1
            # Flat index j*p + k
            b_indices = [
                2 * J * p + 2 * K,       # e (2J, 2K)
                2 * J * p + (2 * K + 1), # f (2J, 2K+1)
                (2 * J + 1) * p + 2 * K,       # g (2J+1, 2K)
                (2 * J + 1) * p + (2 * K + 1)  # h (2J+1, 2K+1)
            ]
            
            # Global indices for C (i, k) -> Trace convention W[k*n + i, r]
            # i=0,1; k=2K, 2K+1
            # Flat index k*n + i
            c_indices = [
                2 * K * n + 0,       # C00 (i=0, k=2K)
                (2 * K + 1) * n + 0, # C01 (i=0, k=2K+1)
                2 * K * n + 1,       # C10 (i=1, k=2K)
                (2 * K + 1) * n + 1  # C11 (i=1, k=2K+1)
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
                        
    return {"U": U, "V": V, "W": W}
