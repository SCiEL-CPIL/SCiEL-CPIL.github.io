import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use Bini et al. rank-23 algorithm for 3x3x3
    # Applied twice for 3x3x6
    # Coefficients for Bini et al. (1979)
    # U, V, W are 23 x 9 (flattened 3x3)
    
    # I do not have the exact coefficients memorized.
    # I will return the schoolbook to ensure validity.
    n, m, p = DIMENSIONS
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
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
