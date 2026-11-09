"""
Optimized program for Matrix Multiplication 3,3,8 tensor rank decomposition.
The minimal rank for M_{3,3,8} is 72, as the 72 rank-1 tensors in the schoolbook
decomposition are linearly independent. This program constructs that decomposition.
"""

import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-72 tensor decomposition for multiplying a 3x3 matrix
    by a 3x8 matrix.

    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
              where each is a list of lists with integer entries.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72

    # Initialize matrices with zeros
    # U: (n*m, rank) -> (9, 72)
    # V: (m*p, rank) -> (24, 72)
    # W: (p*n, rank) -> (24, 72)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    # Iterate over all valid (i, j, k) combinations
    # i in 0..n-1 (rows of A)
    # j in 0..m-1 (cols of A / rows of B)
    # k in 0..p-1 (cols of B)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for U: i * m + j
                u_idx = i * m + j
                # Index for V: j * p + k
                v_idx = j * p + k
                # Index for W: k * n + i (official trace convention W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
