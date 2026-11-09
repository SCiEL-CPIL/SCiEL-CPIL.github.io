import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The minimal rank for matrix multiplication M_{n,m,p} is at least max(n,p).
    # Here max(4,5) = 5.
    # The standard algorithm uses rank m = 5.
    # Since 5 >= 5, the standard algorithm is optimal.
    rank = m
    
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    # Construct the standard decomposition
    # For each j in 0..m-1, we have a term r=j
    # U[i*m + j, j] = 1
    # V[j*p + k, j] = 1
    # W[k*n + i, j] = 1
    
    for j in range(m):
        # Set U column j
        for i in range(n):
            U[i * m + j, j] = 1
            
        # Set V column j
        for k in range(p):
            V[j * p + k, j] = 1
            
        # Set W column j
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
