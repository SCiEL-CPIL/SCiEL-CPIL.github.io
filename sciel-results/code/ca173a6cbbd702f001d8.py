"""Matrix multiplication 2x4x5 - rank-38 exact tensor decomposition via Block-Strassen."""

import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-38 decomposition of 2x4x5 matrix multiplication.
    
    Split m=4 into two blocks of m=2. Each 2x2x5 sub-problem is split into:
      - 2x2x2 (Strassen, rank 7)
      - 2x2x3 (schoolbook, rank 12)
    Total: 2 * (7 + 12) = 38.
    """
    # Strassen rank-7 for 2x2x2
    strassen_U = [
        [0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, -1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, -1],
    ]
    strassen_V = [
        [1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1, 1, 1, -1],
        [-1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, -1, 0, 0],
    ]
    
    # Schoolbook 2x2x3 (rank 12)
    U_223 = [[0]*12 for _ in range(4)]
    V_223 = [[0]*12 for _ in range(6)]
    W_223 = [[0]*12 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_223[i*2+j][col] = 1
                V_223[j*3+k][col] = 1
                W_223[k*2+i][col] = 1
                col += 1
    
    n, m, p = 2, 4, 5
    total_rank = 38
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for block in range(2):
        # Strassen for 2x2x2 (first 2 cols of B in this block)
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    row_u = i * m + (block * 2 + j)
                    U_full[row_u][col_offset + r] = strassen_U[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    row_v = (block * 2 + j) * p + k
                    V_full[row_v][col_offset + r] = strassen_V[j*2+k][r]
            for i in range(2):
                for k in range(2):
                    row_w = k * n + i
                    W_full[row_w][col_offset + r] = strassen_W[k*2+i][r]
        col_offset += 7
        
        # Schoolbook for 2x2x3 (last 3 cols of B in this block)
        for r in range(12):
            for i in range(2):
                for j in range(2):
                    row_u = i * m + (block * 2 + j)
                    U_full[row_u][col_offset + r] = U_223[i*2+j][r]
            for j in range(2):
                for k in range(3):
                    row_v = (block * 2 + j) * p + (2 + k)
                    V_full[row_v][col_offset + r] = V_223[j*3+k][r]
            for i in range(2):
                for k in range(3):
                    row_w = (2 + k) * n + i
                    W_full[row_w][col_offset + r] = W_223[k*2+i][r]
        col_offset += 12
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
