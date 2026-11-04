"""Tensor decomposition for 3x4 * 4x7 matrix multiplication."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    The output is a dict with keys 'U', 'V', 'W' where:
      U has shape (3*4, rank) = (12, rank)
      V has shape (4*7, rank) = (28, rank)
      W has shape (7*3, rank) = (21, rank)
    
    The decomposition satisfies:
      For all i in [0,3), j in [0,4), k in [0,7):
        sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*3+i, r] = 1
      For all other (a, b, c) tuples:
        sum_r U[a, r] * V[b, r] * W[c, r] = 0
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # Use schoolbook decomposition with rank = n*m*p = 84
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
