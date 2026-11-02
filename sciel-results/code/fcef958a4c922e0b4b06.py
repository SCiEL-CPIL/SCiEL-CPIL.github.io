"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-36 schoolbook decomposition for 2x2 * 2x9 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    U has shape (n*m, rank) = (4, 36): U[i*m+j, r] selects A[i,j]
    V has shape (m*p, rank) = (18, 36): V[j*p+k, r] selects B[j,k]
    W has shape (p*n, rank) = (18, 36): W[k*n+i, r] selects C[i,k]
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = n * m * p      # rank = 36
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 36
    V = [[0] * rank for _ in range(m * p)]   # 18 x 36
    W = [[0] * rank for _ in range(p * n)]   # 18 x 36
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term: A[i,j] * B[j,k] contributes to C[i,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
