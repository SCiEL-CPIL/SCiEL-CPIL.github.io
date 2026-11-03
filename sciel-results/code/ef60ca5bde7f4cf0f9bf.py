"""Program for 3x4x7 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    
    Strategy: Use the schoolbook decomposition with rank = 3*4*7 = 84.
    This is guaranteed correct and within time limits.
    
    For each (i, j, k) where i in [0,3), j in [0,4), k in [0,7):
        U[i*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + i][col] = 1
        col += 1
    """
    n, m, p = 3, 4, 7
    rank = n * m * p  # 84
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 84
    V = [[0] * rank for _ in range(m * p)]  # 28 x 84
    W = [[0] * rank for _ in range(p * n)]  # 21 x 84
    
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
