"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    
    # Decompose M_{2,2,15} into 6 blocks of M_{2,2,2} (rank 7) and 1 block of M_{2,2,1} (rank 2)
    # Total rank = 6 * 7 + 2 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen matrices for 2x2x2
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook matrices for 2x2x1
    U_school = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    V_school = [
        [1, 0],
        [0, 1]
    ]
    W_school = [
        [1, 0],
        [0, 1]
    ]
    
    # Fill Strassen blocks (0 to 5)
    for block in range(6):
        k0 = 2 * block
        k1 = 2 * block + 1
        col_offset = 7 * block
        for r in range(7):
            c = col_offset + r
            # U: 2x2 input, index i*2+j
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            # V: 2x2 input (columns k0, k1), index j*2+k
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            # W: 2x2 output (rows k0, k1), index k*2+i
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
    
    # Fill Schoolbook block (column 14)
    k_last = 14
    col_offset = 42
    for r in range(2):
        c = col_offset + r
        # U: 2x2 input
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        # V: 2x1 input (column 14), index j
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        # W: 2x1 output (row 14), index i
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
