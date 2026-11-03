import time
import json

def construct():
    """
    Returns the schoolbook tensor decomposition for 3x5 by 5x7 matrix multiplication.
    Rank is 3 * 5 * 7 = 105.
    """
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    # Initialize matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                # V row index: j*p + k
                # W row index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
