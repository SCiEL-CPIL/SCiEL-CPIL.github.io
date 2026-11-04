"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = m: one rank-1 term per value of the shared index j
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        # U: rows i*m + j for all i get 1 in column j
        for i in range(n):
            U[i * m + j][j] = 1
        # V: rows j*p + k for all k get 1 in column j
        for k in range(p):
            V[j * p + k][j] = 1
        # W: all rows get 1 in column j
        for idx in range(p * n):
            W[idx][j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
