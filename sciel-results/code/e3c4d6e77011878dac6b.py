"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Use the naive decomposition (rank = n*m*p) which is guaranteed to be a valid
    # decomposition of the all-ones tensor. If the rank 1 solution was rejected,
    # this provides a robust fallback that explicitly constructs the tensor terms.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
