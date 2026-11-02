"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition for 4x5 * 5x5 matrix multiplication.
    Rank = 4*5*5 = 100.
    
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    U[i*m + j][col] = 1  -> selects A[i,j]
    V[j*p + k][col] = 1  -> selects B[j,k]
    W[k*n + i][col] = 1  -> accumulates into C[i,k]
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p  # 100
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
