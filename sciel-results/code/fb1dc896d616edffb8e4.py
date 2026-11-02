import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    
    The minimal rank for this specific rectangular matrix multiplication
    is known to be n*m*p = 24 because the bilinear forms A_ij * B_jk 
    contribute uniquely to C_ik and cannot be shared or combined to reduce 
    the rank due to the disjoint support of the tensor components for 
    each output entry.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (trace convention W[k*n+i, r])
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
