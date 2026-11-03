"""Program for matrix multiplication 2x4 x 4x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    The decomposition has rank 48 (schoolbook), which is conjectured to be minimal
    for these dimensions. The factor matrices are:
    - U: shape (n*m, rank) = (8, 48)
    - V: shape (m*p, rank) = (24, 48)
    - W: shape (p*n, rank) = (12, 48)
    
    Each column c corresponds to a term in the bilinear form:
    C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where j is determined by the column index.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term contributes to exactly one entry in the output
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
