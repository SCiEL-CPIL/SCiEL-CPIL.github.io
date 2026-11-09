"""Matrix multiplication 3,4,8 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 8]

import time

def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    
    The decomposition is of the form:
    C[i, k] = sum_t U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t]
    
    For the schoolbook algorithm, the rank is n*m*p = 3*4*8 = 96.
    Each term in the sum corresponds to a single multiplication A[i, j] * B[j, k].
    
    U shape: (n*m, rank) = (12, 96)
    V shape: (m*p, rank) = (32, 96)
    W shape: (p*n, rank) = (24, 96)
    
    Entries are 1.0 for the matching index, 0.0 otherwise.
    """
    n, m, p = DIMENSIONS
    
    # Calculate schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
