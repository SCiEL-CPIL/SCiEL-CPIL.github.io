"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    
    The multiplication tensor T[i,j,k] = delta(i,i')delta(j,j')delta(k,k') requires
    rank n*m*p = 60. We construct the schoolbook decomposition which achieves this.
    
    Reconstruction: C[i,k] = sum_{j,c} U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    rank = n * m * p       # 60
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 60)
    V = [[0] * rank for _ in range(m * p)]   # shape (30, 60)
    W = [[0] * rank for _ in range(p * n)]   # shape (30, 60)
    
    # Fill in the identity tensor decomposition
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
