"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        # For each column k, we perform a 2x2 by 2x1 multiplication
        # which requires 4 terms (rank 4 for 2x2x2 matmul)
        
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
