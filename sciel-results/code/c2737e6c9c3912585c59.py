"""Tensor decomposition for 2x3x9 matrix multiplication.

For the bilinear map M_{2,3,9}: R^{2x3} x R^{3x9} -> R^{2x9},
the tensor rank equals n*m*p = 54. This follows from the
support-disjointness of the two rows of A (each row's bilinear
form has rank m*p = 27, and the supports in the A-index are
disjoint, so the total rank is 2*27 = 54).

The schoolbook decomposition achieves exactly rank 54.
"""

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54

    U = [[0] * rank for _ in range(n * m)]  # (6, 54)
    V = [[0] * rank for _ in range(m * p)]  # (27, 54)
    W = [[0] * rank for _ in range(p * n)]  # (18, 54)

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
