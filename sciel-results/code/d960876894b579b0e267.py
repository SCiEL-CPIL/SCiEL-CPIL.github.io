"""Program for matrix multiplication tensor rank minimization."""

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-64 exact tensor decomposition for multiplying
    a 2x2 matrix A by a 2x16 matrix B.
    
    The decomposition uses the schoolbook algorithm, which corresponds
    to the trivial rank decomposition of the tensor.
    Rank = n * m * p = 2 * 2 * 16 = 64.
    
    U is (4, 64), V is (32, 64), W is (32, 64).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
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
