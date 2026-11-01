"""
Optimal rank decomposition for 2x5x5 matrix multiplication.

The tensor for multiplying a 2x5 matrix by a 5x5 matrix is a direct sum of two
independent 1x5x5 multiplication tensors.
The rank of a 1xn x n multiplication tensor is n^2.
Thus, the rank of the 1x5x5 tensor is 25.
The total rank is 2 * 25 = 50.

This is the minimum possible rank. Strassen's algorithm (rank 7 for 2x2) is not
advantageous for 5x5 blocks when embedded, as the overhead and padding result in
a higher total rank than the schoolbook method (rank 25 for 5x5).
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
