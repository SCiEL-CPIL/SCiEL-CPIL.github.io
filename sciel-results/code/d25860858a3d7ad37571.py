import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:9]
    # This is 2x2x9. Decomposed into three 2x2x3 blocks.
    # Each 2x2x3 is decomposed into three 2x2x1 blocks.
    # Each 2x2x1 has rank 2.
    # Total 18 terms.
    
    # We iterate over the 3 blocks of B (cols 0-2, 3-5, 6-8)
    # For each block, we iterate over the 2 rows of B (j=0,1)
    # For each row, we iterate over the 2 cols of the block (k=0,1,2)
    # Wait, 2x2x1 means A(2x2), B(2x1), C(2x1).
    # The term is A[i,j] * B[j,k] -> C[i,k].
    # For a fixed k, we have 2 terms (j=0 and j=1).
    
    # Loop over k_block (0, 1, 2)
    for kb in range(3):
        # Loop over k_offset (0, 1, 2)
        for ko in range(3):
            k = kb * 3 + ko
            # Term for j=0
            # A[i,0] * B[0,k] -> C[i,k]
            # U[i*2+0, col] = 1
            # V[0*9+k, col] = 1
            # W[k*2+i, col] = 1
            for i in range(2):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1
            
            # Term for j=1
            # A[i,1] * B[1,k] -> C[i,k]
            # U[i*2+1, col] = 1
            # V[1*9+k, col] = 1
            # W[k*2+i, col] = 1
            for i in range(2):
                U[i * m + 1][col] = 1
                V[1 * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[0:2, 2] * B[2, 0:9]
    # This is 2x1x9. Rank 9.
    # Term: A[i,2] * B[2,k] -> C[i,k]
    # For each k, one term.
    for k in range(9):
        for i in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
