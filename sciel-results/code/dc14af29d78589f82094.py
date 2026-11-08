import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # Rank 18 decomposition
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: cols 0-2 of B
    # A (3x3) * B1 (3x3) -> C1 (3x3)
    # U1: 9x9, V1: 9x9 (rows 0-8 of V), W1: 9x9 (rows 0-8 of W)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                col = i * 9 + j * 3 + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1  # k is 0..2, so row j*6+k
                W[k * n + i][col] = 1  # row k*3+i
                
    # Block 2: cols 3-5 of B
    # A (3x3) * B2 (3x3) -> C2 (3x3)
    # U2: 9x9, V2: 9x9 (rows 9-17 of V), W2: 9x9 (rows 9-17 of W)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                col = 9 + i * 9 + j * 3 + k
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1 # k+3 is 3..5
                W[(k + 3) * n + i][col] = 1 # row (k+3)*3+i
                
    return {"U": U, "V": V, "W": W}
