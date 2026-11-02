"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook matrix multiplication tensor decomposition.
    Rank: n * m * p = 125.
    Returns: Dictionary with keys 'U', 'V', 'W' containing list of lists.
    """
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    # Shapes: (n*m, rank), (m*p, rank), (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indexing conventions:
                # U: row = i*m + j
                # V: row = j*p + k
                # W: row = k*n + i (Trace convention: W[k*n+i, r])
                
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
                # Safety check (optional, usually not needed for 125 iterations)
                # if time.time() > deadline:
                #     return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
