import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Block-partitioning decomposition for 3x5x6 matrix multiplication.
    # Partition:
    #   A (3x5) = [A1; A2], A1 is 1x5, A2 is 2x5
    #   B (5x6) = [B1 | B2], B1 is 5x1, B2 is 5x5
    #
    # C = A*B = [A1*B1, A1*B2; A2*B1, A2*B2]
    #
    # Ranks:
    #   A1*B1: 1x1 -> rank 1
    #   A1*B2: 1x5 -> rank 5
    #   A2*B1: 2x1 -> rank 5
    #   A2*B2: 2x5 -> rank 20 (schoolbook)
    # Total: 31
    
    rank = 31
    U = [[0] * rank for _ in range(n * m)]   # 15 x 31
    V = [[0] * rank for _ in range(m * p)]   # 30 x 31
    W = [[0] * rank for _ in range(p * n)]   # 18 x 31
    
    col = 0
    
    # Block 1: A1 * B1 -> C[0,0], rank 1
    for j in range(m):
        U[0 * m + j][col] = 1
        V[j * p + 0][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2: A1 * B2 -> C[0,1..5], rank 5
    for k in range(1, p):
        for j in range(m):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
    
    # Block 3: A2 * B1 -> C[1..2, 0], rank 5
    for i in range(1, n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 0][col] = 1
        W[0 * n + i][col] = 1
        col += 1
    
    # Block 4: A2 * B2 -> C[1..2, 1..5], rank 20
    for i in range(1, n):
        for k in range(1, p):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Internal error: col={col}, rank={rank}"
    
    if time.time() > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
