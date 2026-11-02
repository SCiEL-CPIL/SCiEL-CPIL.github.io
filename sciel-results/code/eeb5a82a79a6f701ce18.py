"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-m minimal decomposition: one rank-1 term per inner dimension index j
    # For each j in [0, m), the term covers all (i, k) pairs with:
    #   U[i*m + j, j] = 1
    #   V[j*p + k, j] = 1
    #   W[k*n + i, j] = 1
    rank = m  # = 5
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
