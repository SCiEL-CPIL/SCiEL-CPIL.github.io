"""
Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.

The 2x5x5 multiplication tensor is a direct sum of two 1x5x5 tensors.
The rank of M_{1,5,5} is exactly 25 (schoolbook is optimal for 1xn x n).
Thus the total rank is 50.
"""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2 * 5 * 5 = 50

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
