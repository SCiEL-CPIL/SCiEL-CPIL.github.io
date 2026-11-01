"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-m*p decomposition: one term per (j,k) pair
    # Term c = j*p+k: V[j*p+k,c]=1, U[i*m+j,c]=1 for all i, W[k*n+i,c]=1 for all i
    # This works because V isolates the (j,k) pair, and U,W provide the i-dependence.
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            V[j * p + k][c] = 1
            for i in range(n):
                U[i * m + j][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
