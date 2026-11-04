import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x5 matrix multiplication.
    This implements the standard schoolbook algorithm which has rank n*m*p = 75.
    The factor matrices U, V, W are constructed such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i, i'} delta_{j, j'} delta_{k, k'}
    (effectively reconstructing the multiplication tensor).
    
    The decomposition is exact and uses integer entries (0 or 1).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Construct the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i of A, column j -> i*m + j
                U[i * m + j, col] = 1
                
                # V index: row j of B, column k -> j*p + k
                V[j * p + k, col] = 1
                
                # W index: k*n + i
                W[k * n + i, col] = 1
                col += 1
                
    # Check time
    if time.time() > deadline:
        pass

    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
