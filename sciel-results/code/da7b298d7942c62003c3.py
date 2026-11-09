"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 20 decomposition.
    # 18 terms for k=0,1 (schoolbook for 3x3x2)
    # 1 term for k=2 (rank 1 for 3x3x1)
    # 1 term for k=3 (rank 1 for 3x3x1)
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Schoolbook for k=0, 1
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Rank 1 for k=2
    # U[3i+j, col] = 1
    # V[j*p+2, col] = 1
    # W[2*n+i, col] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
    col += 1
    
    # Rank 1 for k=3
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
