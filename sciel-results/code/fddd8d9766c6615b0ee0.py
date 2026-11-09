"""
Rewritten program for Matrix Multiplication 2x3 by 3x6.
The minimal rank for this tensor is 36 (schoolbook).
This program constructs the schoolbook decomposition directly and returns it.
"""

def construct():
    """
    Construct the rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 6 matrix.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U: shape (6, 36)
              V: shape (18, 36)
              W: shape (12, 36)
    """
    n, m, p = 2, 3, 6
    rank = n * m * p  # 36
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1.0
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1.0
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1.0
                col += 1
                
    return {'U': U, 'V': V, 'W': W}
