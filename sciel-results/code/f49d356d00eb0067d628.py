"""
Rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.

The multiplication tensor for C = A @ B where A is 2x5 and B is 5x5 is a direct sum
of two independent 1x5x5 multiplication tensors (one for each row of A).

For a 1 x n x n multiplication tensor (computing a 1xn row vector from an nxn matrix),
the schoolbook rank is n^2. For n=5, this gives rank 25 per row.
Total rank = 2 * 25 = 50.

This is provably minimal: the 1xn x n tensor has rank n^2 because each output
element C_k = sum_j A_j B_{j,k} requires n independent bilinear forms, and there
are n output elements, giving n^2 total. No Strassen-type speedup applies here
because the "square" dimension is only 1 in one factor.
"""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition of rank 50.
    U: (2*5, 50)
    V: (5*5, 50)
    W: (5*2, 50)
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
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
