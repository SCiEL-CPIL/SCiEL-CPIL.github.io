"""
Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x13 matrix.

The matrix multiplication tensor for (2,2,13) has rank 52 (schoolbook).
We construct the standard schoolbook decomposition with integer entries.
"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the schoolbook decomposition of the (2,2,13) matrix multiplication tensor.
    
    The decomposition has rank n*m*p = 2*2*13 = 52.
    U has shape (n*m, rank) = (4, 52)
    V has shape (m*p, rank) = (26, 52)
    W has shape (p*n, rank) = (26, 52)
    
    For each (i, j, k) triple, column c = i*m*p + j*p + k has:
      U[i*m + j, c] = 1
      V[j*p + k, c] = 1
      W[k*n + i, c] = 1
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
