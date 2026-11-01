"""Matrix multiplication 2,3,10 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 10]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank = 60
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 60)
    V = [[0] * rank for _ in range(m * p)]  # shape (30, 60)
    W = [[0] * rank for _ in range(p * n)]  # shape (20, 60)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Check if we're approaching deadline
    if time.time() > deadline:
        pass  # We're done, return immediately
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
