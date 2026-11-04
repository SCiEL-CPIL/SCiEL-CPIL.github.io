"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-36 schoolbook decomposition of the 3x3x4 matrix multiplication tensor.
    
    The decomposition is exact: for all valid indices,
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k}
    where delta is 1 if (i,j,k) corresponds to a valid multiplication term and 0 otherwise.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 3 * 4 = 36
    
    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
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
