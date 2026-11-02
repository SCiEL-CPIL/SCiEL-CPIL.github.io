"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Schoolbook decomposition for 2x4 * 4x7 -> 2x7 matrix multiplication.
    
    Rank = n*m*p = 2*4*7 = 56.
    U shape: (n*m, rank) = (8, 56)
    V shape: (m*p, rank) = (28, 56)
    W shape: (p*n, rank) = (14, 56)
    
    Convention: W[k*n + i, r] corresponds to output C[i,k]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] contributes to C[i,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
