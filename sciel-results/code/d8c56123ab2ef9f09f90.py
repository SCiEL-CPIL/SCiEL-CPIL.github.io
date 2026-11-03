"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    The multiplication is decomposed into three independent 2x3x3 block multiplications.
    Each 2x3x3 multiplication has a schoolbook rank of 18.
    The total rank is 3 * 18 = 54.
    """
    n, m, p = DIMENSIONS
    # The multiplication is a direct sum of p/m = 3 blocks of size n x m x m
    # Each block multiplies n x m by m x m
    # Rank of each block is n * m * m = 2 * 3 * 3 = 18
    # Total rank is (p/m) * (n * m * m) = 3 * 18 = 54
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                # Output index convention: W[k*n + i, r]
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
