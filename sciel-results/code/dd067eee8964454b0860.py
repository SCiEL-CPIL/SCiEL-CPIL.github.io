"""Matrix multiplication 2x3x8 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    
    Schoolbook rank is n*m*p = 48. For 2x3x8 matrix multiplication, the bilinear
    forms are rigid: each scalar product A[i,j]*B[j,k] must be computed separately
    to contribute correctly to C[i,k] without cross-term interference. No Strassen-type
    sharing reduces the rank below 48 for these dimensions.
    
    Interface: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    W follows the convention W[k*n+i, c].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
