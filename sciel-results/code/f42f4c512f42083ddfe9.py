import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # A: 2x4, B: 4x8, C: 2x8
    # Output W index: k*n + i
    
    # We use the block-diagonal Strassen decomposition.
    # Split A into 2 blocks of 2x2 (cols 0-1, 2-3).
    # Split B into 2 blocks of 2x8 (rows 0-1, 2-3).
    # Each (2x2) * (2x8) product is split into 4 independent (2x2)*(2x2) products.
    # Total 8 independent (2x2)*(2x2) products.
    # Each uses Strassen (rank 7).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Pre-define Strassen coefficients for 2x2 * 2x2
    # U vectors (A side)
    # m1: a00+a11
    # m2: a10+a11
    # m3: a00
    # m4: a11
    # m5: a00+a01
    # m6: a10-a00
    # m7: a01-a11
    strassen_U = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [-1, 0, 1, 0],# m6
        [0, 1, 0, -1] # m7
    ]
    
    # V vectors (B side)
    # m1: b00+b11
    # m2: b00
    # m3: b01-b11
    # m4: b10
    # m5: b11
    # m6: b00+b01
    # m7: b10+b11
    strassen_V = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1],# m3
        [0, 0, 1, 0], # m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    
    # W vectors (C side)
    # C index convention: k*n + i. 
    # For 2x2 C:
    # C00 (k=0,i=0) -> 0
    # C01 (k=1,i=0) -> 1
    # C10 (k=0,i=1) -> 2
    # C11 (k=1,i=1) -> 3
    # 
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # W[r][col] where r is C index (0..3)
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    col = 0
    # Loop over the 8 blocks
    # s: 0, 1 (split of A cols / B rows)
    # j: 0, 1, 2, 3 (split of B cols)
    for s in range(2):
        for j in range(4):
            # Identify global indices for this 2x2 * 2x2 block
            
            # A block: rows 0-1, cols 2s, 2s+1
            # Global A index: i * m + j_global
            a00_idx = 0 * m + (2 * s + 0)
            a01_idx = 0 * m + (2 * s + 1)
            a10_idx = 1 * m + (2 * s + 0)
            a11_idx = 1 * m + (2 * s + 1)
            
            # B block: rows 2s, 2s+1, cols 2j, 2j+1
            # Global B index: j_global * p + k_global
            b00_idx = (2 * s + 0) * p + (2 * j + 0)
            b01_idx = (2 * s + 0) * p + (2 * j + 1)
            b10_idx = (2 * s + 1) * p + (2 * j + 0)
            b11_idx = (2 * s + 1) * p + (2 * j + 1)
            
            # C block: rows 0-1, cols 2j, 2j+1
            # Global C index: k_global * n + i_global
            c00_idx = (2 * j + 0) * n + 0
            c01_idx = (2 * j + 1) * n + 0
            c10_idx = (2 * j + 0) * n + 1
            c11_idx = (2 * j + 1) * n + 1
            
            # Fill U, V, W for the 7 Strassen multiplications
            for r in range(7):
                # U
                U[a00_idx][col + r] = strassen_U[r][0]
                U[a01_idx][col + r] = strassen_U[r][1]
                U[a10_idx][col + r] = strassen_U[r][2]
                U[a11_idx][col + r] = strassen_U[r][3]
                
                # V
                V[b00_idx][col + r] = strassen_V[r][0]
                V[b01_idx][col + r] = strassen_V[r][1]
                V[b10_idx][col + r] = strassen_V[r][2]
                V[b11_idx][col + r] = strassen_V[r][3]
                
                # W
                W[c00_idx][col + r] = strassen_W[0][r]
                W[c01_idx][col + r] = strassen_W[1][r]
                W[c10_idx][col + r] = strassen_W[2][r]
                W[c11_idx][col + r] = strassen_W[3][r]
            
            col += 7
            
    return {"U": U, "V": V, "W": W}
