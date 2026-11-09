import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the p=8 dimension into two parts of 4.
    # Compute 3x4 multiplication twice.
    # 3x4 is split into 3x3 and 3x1.
    # 3x3 is schoolbook (rank 9).
    # 3x1 is schoolbook (rank 9).
    # Total rank per 3x4 block: 18.
    # Total rank: 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 9 x 36
    V = [[0] * rank for _ in range(m * p)] # 24 x 36
    W = [[0] * rank for _ in range(p * n)] # 24 x 36
    
    col = 0
    # Block 1: Columns 0-3 of B
    # 3x3 part: B[:, 0:3]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # 3x1 part: B[:, 3]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1 # Output col 3
            col += 1
            
    # Block 2: Columns 4-7 of B
    # 3x3 part: B[:, 4:7]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + 4 + k][col] = 1
                W[(4 + k) * n + i][col] = 1
                col += 1
    # 3x1 part: B[:, 7]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 7][col] = 1
            W[7 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
