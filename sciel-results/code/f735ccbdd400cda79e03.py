"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    
    We use the schoolbook decomposition which gives rank n*m*p = 60.
    The multiplication tensor decomposes as sum over (i,j,k) of rank-1 terms.
    
    Reconstruction: C[i,k] = sum_{c} U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    where the sum is over all c (rank dimension), and for each c there is exactly one
    (i,j,k) tuple such that U[i*m+j,c]=V[j*p+k,c]=W[k*n+i,c]=1.
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    rank = n * m * p       # 60
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 60)
    V = [[0] * rank for _ in range(m * p)]   # shape (30, 60)
    W = [[0] * rank for _ in range(p * n)]   # shape (30, 60)
    
    # Assign each rank-1 term to a column
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
