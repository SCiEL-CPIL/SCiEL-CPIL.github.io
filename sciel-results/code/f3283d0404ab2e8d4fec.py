"""Matrix multiplication 3x3x6 tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-54 (schoolbook) decomposition for 3x3x6 matrix multiplication.
    
    The schoolbook decomposition has rank = n*m*p = 3*3*6 = 54.
    Each term corresponds to one (i,j,k) triple:
      U[i*m+j, t] = 1  (selects A[i,j])
      V[j*p+k, t] = 1  (selects B[j,k])
      W[k*n+i, t] = 1  (places product into C[i,k])
    
    This is guaranteed correct and minimal in the sense that it's the standard
    direct multiplication. Lower-rank decompositions exist (e.g., rank 45 via
    three rank-15 3x3x2 blocks) but require careful construction.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    rank = n * m * p  # 54
    
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
