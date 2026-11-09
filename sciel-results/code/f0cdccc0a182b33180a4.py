import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x5 matrix.
    
    Output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes
    (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries
    that exactly decompose the matrix multiplication tensor.
    
    Score is the rank (lower is better).
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = n * m * p  # 75
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Schoolbook decomposition
    # Each term r corresponds to a unique triple (i, j, k)
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    # Verification (optional, but good for confidence)
    # The tensor T has T[u_idx, v_idx, w_idx] = 1 if (u_idx, v_idx, w_idx) 
    # corresponds to a valid (i,j,k) triple, 0 otherwise.
    # Our decomposition sum_r U[:,r] V[:,r]^T W[:,r]^T should equal T.
    # Actually, the tensor is an array of shape (n*m, m*p, p*n).
    # Let's verify a few random points.
    # But for the sake of speed, we trust the schoolbook.
    
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
