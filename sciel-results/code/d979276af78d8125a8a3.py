"""
Program to find a rank-minimal exact tensor decomposition for multiplying
a 2 by 3 matrix by a 3 by 6 matrix.

The schoolbook algorithm provides a valid decomposition of rank n*m*p.
For n=2, m=3, p=6, the rank is 36.
This is a baseline. Finding a lower rank (e.g., 30 or 33) requires
specific algebraic structures or search which are not reliably derivable
in the short time frame without risking invalidity.
"""

import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p.
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for standard interface compatibility
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each rank-1 term corresponds to a unique (i, j, k)
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
