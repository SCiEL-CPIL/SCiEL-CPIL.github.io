import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays of zeros
    # Shapes: (n*m, rank), (m*p, rank), (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j, col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k, col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
                # Check time
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
            
    # Convert to lists of lists if necessary? 
    # The interface says "factor matrices". Numpy arrays are usually accepted.
    # But to be safe, let's return numpy arrays.
    return {"U": U, "V": V, "W": W}
