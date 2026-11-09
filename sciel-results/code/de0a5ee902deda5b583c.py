"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    # Rank is m=4
    # Decomposition:
    # For each r in 0..m-1:
    # U[i*m + j, r] = 1 if j == r else 0
    # V[j*p + k, r] = 1 if j == r else 0
    # W[k*n + i, r] = 1
    
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # Fill U: column r has 1s where j == r
        # Index is i*m + j
        for i in range(n):
            for j in range(m):
                if j == r:
                    U[i * m + j][r] = 1
        
        # Fill V: column r has 1s where j == r
        # Index is j*p + k
        for j in range(m):
            for k in range(p):
                if j == r:
                    V[j * p + k][r] = 1
        
        # Fill W: column r has 1s everywhere
        # Index is k*n + i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
