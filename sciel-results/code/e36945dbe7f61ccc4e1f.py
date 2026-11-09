"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for matrix multiplication M_{n,m,p} is m.
    # We construct a rank-m decomposition.
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m + j, alpha] = 1 if j == alpha else 0
    for i in range(n):
        for j in range(m):
            alpha = j
            U[i * m + j][alpha] = 1
            
    # V[j*p + k, alpha] = 1 if j == alpha else 0
    for j in range(m):
        for k in range(p):
            alpha = j
            V[j * p + k][alpha] = 1
            
    # W[k*n + i, alpha] = 1 for all k, i, alpha
    for k in range(p):
        for i in range(n):
            for alpha in range(rank):
                W[k * n + i][alpha] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
