import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    timeout = 120.0
    
    # Hypothesis: The tensor T_{i, j, k} = 1 is rank 1.
    rank = 1
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 1)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 1)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 1)
    
    # Set all entries to 1
    for i in range(n * m):
        U[i][0] = 1
    for j in range(m * p):
        V[j][0] = 1
    for k in range(p * n):
        W[k][0] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
