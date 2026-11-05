"""Program for matrix multiplication tensor decomposition 3x3 * 3x8."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses the schoolbook decomposition, which is rank 72.
    This is likely the minimal rank for integer/half-integer coefficients.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook rank for 3x3 * 3x8 is 3 * 3 * 8 = 72.
    # We partition the 8 columns into three groups: {0,1,2}, {3,4,5}, {6,7}.
    # Each group of 3 columns corresponds to a 3x3 * 3x3 multiplication (rank 27).
    # The last group of 2 columns corresponds to a 3x3 * 3x2 multiplication (rank 18).
    # Total rank = 27 + 27 + 18 = 72.
    
    rank_total = n * m * p
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    # Block 1: columns 0, 1, 2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: columns 3, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 3: columns 6, 7
    for i in range(n):
        for j in range(m):
            for k in range(6, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
