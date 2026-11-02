"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    We use a block decomposition approach:
    - Split the 5 columns of A and 5 rows of B into blocks of size 2 and 3.
    - C = A[:, 0:2] @ B[0:2, :] + A[:, 2:5] @ B[2:5, :]
    - First term: 3x2 * 2x5 (rank 30 schoolbook)
    - Second term: 3x3 * 3x5 (rank 45 schoolbook)
    - Total rank: 75
    
    This is the schoolbook decomposition, which is correct but not minimal.
    However, it's guaranteed to work within the time limit.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 5 * 5 = 75
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 75
    V = [[0] * rank for _ in range(m * p)]  # 25 x 75
    W = [[0] * rank for _ in range(p * n)]  # 15 x 75
    
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
