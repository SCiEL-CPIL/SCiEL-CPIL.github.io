"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Default: Schoolbook decomposition
    # Rank = n * m * p
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
                
    # No effective search strategy found for rank < 105 for 3x5x7 within constraints
    # Returning schoolbook is the only guaranteed valid solution.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
