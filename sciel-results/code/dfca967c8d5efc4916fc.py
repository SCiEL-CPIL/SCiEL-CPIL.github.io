"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Decompose 3x4 into 3x3 and 3x1
    # Part 1: 3x3 multiplication (A[0:3, 0:3] * B[0:3, 0:3])
    # Part 2: 3x1 multiplication (A[0:3, 0:3] * B[0:3, 3:4])
    
    # Rank for 3x3 schoolbook is 27
    # Rank for 3x1 schoolbook is 3
    # Total rank 30
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 3x3
    # A is 3x3, B is 3x3 (cols 0,1,2)
    # C is 3x3 (cols 0,1,2)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 3x1
    # A is 3x3, B is 3x1 (col 3)
    # C is 3x1 (col 3)
    # Rank 3 decomposition
    # C_i = sum_j A[i,j] B[j,3]
    # Term r corresponds to j
    for j in range(3):
        for i in range(3):
            # U index: i*m + j
            # V index: j*p + 3
            # W index: 3*n + i
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
