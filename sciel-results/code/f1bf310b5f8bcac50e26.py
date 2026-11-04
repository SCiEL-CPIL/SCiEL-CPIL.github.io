import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # We decompose into:
    # 1. A1 (3x3) * B1a (3x3) -> C1a (3x3). Rank 27.
    # 2. A1 (3x3) * B1b (3x3) -> C1b (3x3). Rank 27.
    # 3. A2 (3x1) * B2 (1x6) -> C2 (3x6). Rank 1.
    # Total rank = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 * B1a
    # A1 is A[0:3, 0:3]
    # B1a is B[0:3, 0:3]
    # Output C1a is C[0:3, 0:3]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: A1 * B1b
    # A1 is A[0:3, 0:3]
    # B1b is B[0:3, 3:6]
    # Output C1b is C[0:3, 3:6]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # B index is 3+k
                U[i * m + j][col] = 1
                V[j * p + (3 + k)][col] = 1
                # Output index k is 3+k in the full 6-col matrix?
                # No, C is 3x6. C[i, 3+k].
                # W index is k_full * n + i.
                # k_full = 3 + k.
                W[(3 + k) * n + i][col] = 1
                col += 1
                
    # Block 3: A2 * B2
    # A2 is A[0:3, 3]
    # B2 is B[3, 0:6]
    # Output C2 is C[0:3, 0:6]
    # C[i, k] = A[i, 3] * B[3, k]
    for i in range(3):
        U[i * m + 3][col] = 1
    for k in range(6):
        V[3 * p + k][col] = 1
    for i in range(3):
        for k in range(6):
            W[k * n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
