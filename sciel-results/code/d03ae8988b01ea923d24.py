import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    
    # Strategy: Split m=3 into 2 and 1.
    # A is 2x3, B is 3x8.
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x8); B2 (1x8)]
    # AB = A1 B1 + A2 B2
    
    # Part 1: A1 B1 (2x2 * 2x8)
    # B1 is 2x8, split into 4 blocks of 2x2.
    # B1 = [B11, B12, B13, B14]
    # A1 B1 = A1 B11 + A1 B12 + A1 B13 + A1 B14
    # Each 2x2*2x2 has rank 7 (Strassen).
    # Total rank for this part: 4 * 7 = 28.
    
    # Part 2: A2 B2 (2x1 * 1x8)
    # Outer product. Rank 16.
    # Total rank: 28 + 16 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)] # 6 x 44
    V = [[0] * rank for _ in range(m * p)] # 24 x 44
    W = [[0] * rank for _ in range(p * n)] # 16 x 44
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Inputs: a11, a12, a21, a22 (indices 0,1,2,3)
    #         b11, b12, b21, b22 (indices 0,1,2,3)
    # Outputs: c11, c12, c21, c22 (indices 0,1,2,3)
    # 7 multiplications:
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b12 - b22)
    # m4 = a22(b21 - b11)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    strassen_U = [
        [1, 1, 1, 0, 1, 0, 0], # a11
        [0, 0, 1, 0, 1, -1, 1], # a12
        [0, 1, 0, 0, 0, 1, 0], # a21
        [1, 1, 0, 1, 0, 0, -1] # a22
    ]
    strassen_V = [
        [1, 1, -1, 1, 0, 1, 1], # b11
        [0, 0, 1, 0, 1, 1, 0], # b12
        [1, 0, 0, -1, 0, 0, 1], # b21
        [1, 0, 0, 0, 1, 0, 1] # b22
    ]
    strassen_W = [
        [1, 0, 0, 0], # c11
        [0, 1, 0, 0], # c12
        [0, 0, 1, 0], # c21
        [0, 0, 0, 1] # c22
    ]
    # Note: W indices for Strassen are relative to the 2x2 output block.
    # We need to map these to the global W indices.
    
    col = 0
    # Part 1: 4 blocks of Strassen
    for block in range(4):
        # A1 indices: 0,1,2,3 (rows 0,1 of A, cols 0,1 of A)
        # B1 indices: block*2, block*2+1 (rows 0,1 of B, cols block*2, block*2+1 of B)
        # Output indices: block*2, block*2+1 (rows 0,1 of C, cols block*2, block*2+1 of C)
        
        for r in range(7):
            # U entries for A1
            U[0][col] = strassen_U[0][r]
            U[1][col] = strassen_U[1][r]
            U[2][col] = strassen_U[2][r]
            U[3][col] = strassen_U[3][r]
            
            # V entries for B1
            # B1 is rows 0,1 of B (global indices 0,1)
            # Cols are block*2, block*2+1
            V[0 + block*2][col] = strassen_V[0][r]
            V[1 + block*2][col] = strassen_V[1][r]
            V[2 + block*2][col] = strassen_V[2][r]
            V[3 + block*2][col] = strassen_V[3][r]
            
            # W entries for Output
            # Output is 2x8.
            # Rows 0,1. Cols block*2, block*2+1.
            # Global W index is k*n + i.
            # k is row of output (0,1). i is col of output (block*2, block*2+1).
            # W[k*n + i]
            
            # c11 -> k=0, i=block*2
            W[0*n + (block*2)][col] = strassen_W[0][r]
            # c12 -> k=0, i=block*2+1
            W[0*n + (block*2+1)][col] = strassen_W[1][r]
            # c21 -> k=1, i=block*2
            W[1*n + (block*2)][col] = strassen_W[2][r]
            # c22 -> k=1, i=block*2+1
            W[1*n + (block*2+1)][col] = strassen_W[3][r]
            
            col += 1
    
    # Part 2: Outer product A2 B2
    # A2 is rows 0,1 of A, col 2 of A. (Global indices 4, 5)
    # B2 is row 2 of B, cols 0..7 of B. (Global indices 16..23)
    # Output is 2x8.
    
    for i in range(n):
        for k in range(p):
            # U[i*m + 2] = 1
            # V[2*p + k] = 1
            # W[k*n + i] = 1
            
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
