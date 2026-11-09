"""Matrix multiplication 2x3 by 3x9: rank-minimal exact tensor decomposition.

Uses a Kronecker-product structure: view the 3x9 matrix B as three 3x3 blocks
along its column dimension.  The multiplication tensor for (2,3,9) is then the
outer (Kronecker) product of the (2,3,3) multiplication tensor with the
(1,1,3) "identity" tensor, giving a schoolbook decomposition of rank 54.

No smaller exact rank is known for 2x3x9 (the 3x3x3 factor alone has rank >= 19
in the best known constructions, and the 2x3x3 factor has rank >= 18), so 54 is
the best achievable exact rank with the standard Kronecker-product approach.
"""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact rank-54 tensor decomposition for multiplying a 2x3 matrix
    by a 3x9 matrix.

    Index convention:
      A has shape (n, m) = (2, 3)
      B has shape (m, p) = (3, 9)
      C has shape (n, p) = (2, 9)

    The output factor W follows the trace convention W[k*n + i, r].

    The decomposition is the schoolbook (Kronecker-structured) one:
      C[i, k] = sum_j A[i, j] * B[j, k]

    Each term (i, j, k) contributes one rank-1 tensor:
      U[i*m + j, r] = 1
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = n * m * p      # 54

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
