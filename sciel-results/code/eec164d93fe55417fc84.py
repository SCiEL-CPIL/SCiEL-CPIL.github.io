"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    We use the schoolbook algorithm, which has rank n*m*p = 120.
    This is the standard bilinear decomposition where each term corresponds
    to a single multiplication A[i, j] * B[j, k] contributing to C[i, k].
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank = n * m * p  # 120
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
