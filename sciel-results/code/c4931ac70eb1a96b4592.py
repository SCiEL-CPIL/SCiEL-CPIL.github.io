"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # One term per j-index
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        c = j
        # U[i*m + j, c] = 1 for all i
        for i in range(n):
            U[i * m + j][c] = 1
        # V[j*p + k, c] = 1 for all k
        for k in range(p):
            V[j * p + k][c] = 1
        # W[k*n + i, c] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
