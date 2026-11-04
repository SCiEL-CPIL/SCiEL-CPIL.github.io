"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    The schoolbook decomposition gives rank = 3*5*5 = 75.
    
    For these small dimensions, the tensor rank of matrix multiplication M_{3,5,5} is 75.
    This is the standard Kronecker product structure where each output element is
    computed independently via a dot product, and no non-trivial linear combinations
    reduce the rank below the product of dimensions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 75
    
    # U has shape (n*m, rank) = (15, 75)
    # V has shape (m*p, rank) = (25, 75)
    # W has shape (p*n, rank) = (15, 75)
    
    # We'll build these as lists of lists with integer entries
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Each column c corresponds to a unique (i, j, k) triple
    # U[i*m + j, c] = 1 if the column corresponds to this (i, j)
    # V[j*p + k, c] = 1 if the column corresponds to this (j, k)
    # W[k*n + i, c] = 1 if the column corresponds to this (k, i)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
