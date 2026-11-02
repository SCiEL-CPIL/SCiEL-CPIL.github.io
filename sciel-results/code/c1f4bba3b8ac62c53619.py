"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 62 decomposition using block partitioning 4x4 + 1
    rank = 62
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Schoolbook 4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
                
    # 2. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4] (a * b^T)
    # a is col 4 of A (i, 4). b is row 4 of B (4, k).
    # Term: A[i, 4] * B[4, k] * W[k, i]
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 5 + i][col] = 1
            col += 1
            
    # 3. A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4] (A44 * b)
    # A[i, j] * B[j, 4] * W[4, i]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1
            
    # 4. A[0:4, 4] * B[4, 4] -> C[0:4, 4] (a * b55)
    # A[i, 4] * B[4, 4] * W[4, i]
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 5 + i][col] = 1
        col += 1
        
    # 5. A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4] (a^T * B44)
    # A[4, j] * B[j, k] * W[k, 4]
    for j in range(4):
        for k in range(4):
            U[4 * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 5 + 4][col] = 1
            col += 1
            
    # 6. A[4, 4] * B[4, 0:4] -> C[4, 0:4] (a55 * b^T)
    # A[4, 4] * B[4, k] * W[k, 4]
    for k in range(4):
        U[4 * 5 + 4][col] = 1
        V[4 * 5 + k][col] = 1
        W[k * 5 + 4][col] = 1
        col += 1
        
    # 7. A[4, 0:4] * B[0:4, 4] -> C[4, 4] (a^T * b)
    # A[4, j] * B[j, 4] * W[4, 4]
    for j in range(4):
        U[4 * 5 + j][col] = 1
        V[j * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
        
    # 8. A[4, 4] * B[4, 4] -> C[4, 4] (a55 * b55)
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
