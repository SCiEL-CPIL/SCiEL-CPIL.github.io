"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Quick verification (optional, but ensures correctness)
    # For rank 50, this is cheap
    # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
    # We know it should be 1 for matching indices, 0 otherwise.
    # We can spot check a few entries.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
