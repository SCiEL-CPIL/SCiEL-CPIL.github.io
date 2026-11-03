"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The matrix multiplication tensor T has T[a, b, c] = 1 iff a = i*m + j, b = j*p + k, c = k*n + i
    for some i in [0, n), j in [0, m), k in [0, p), and 0 otherwise.
    Here n=2, m=3, p=9.
    
    We use the schoolbook decomposition of rank n*m*p = 54.
    For each triple (i, j, k), we create one rank-1 term.
    
    For term r = i*m*p + j*p + k:
      U[i*m + j, r] = 1
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1
    
    This gives (U @ V @ W^T)[a, b, c] = 1 iff there exist i, j, k such that
    a = i*m + j, b = j*p + k, c = k*n + i, which is exactly the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
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
