import time
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication 3x3 by 3x5.
    Rank is n*m*p = 45.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Check if we have time to attempt a more complex search (e.g., Gröbner basis)
    # For now, we return the schoolbook solution which is guaranteed to be valid and fast.
    # A Gröbner basis solver over GF(2) would be computationally infeasible in this context.
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Verify time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # This should never happen for schoolbook
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
