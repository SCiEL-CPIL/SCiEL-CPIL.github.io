"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    """Schoolbook decomposition of rank n*m*p for 6x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 216
    
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
    
    # Check deadline (should never trigger for this O(216) construction)
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
