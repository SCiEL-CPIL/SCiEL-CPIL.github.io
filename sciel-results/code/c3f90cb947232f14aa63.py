import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for 2x5x5 matrix multiplication.
    Returns U, V, W matrices of shape (n*m, rank), (m*p, rank), (p*n, rank).
    The decomposition is exact and uses rank n*m*p.
    """
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill the decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Check time (though this is very fast)
    if time.time() > deadline:
         # If we somehow timed out (unlikely for this size), return what we have
         pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
