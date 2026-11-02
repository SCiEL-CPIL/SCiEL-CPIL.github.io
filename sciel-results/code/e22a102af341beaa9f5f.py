"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = p  # 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        c = k
        # U[3*i+j, c] = 1 for all i, j
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = 1
        # V[j*p+k, c] = 1 for all j
        for j in range(m):
            V[j * p + k][c] = 1
        # W[k*n+i, c] = 1 for all i
        for i in range(n):
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
