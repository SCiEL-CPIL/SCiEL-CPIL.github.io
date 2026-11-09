import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # For 3x4x8, the schoolbook rank is n*m*p = 96.
    # We construct the standard basis decomposition.
    # This corresponds to the Kronecker product of trivial decompositions,
    # or simply the direct sum of rank-1 tensors for each (i,j,k).
    
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Using lists of lists for direct return as required, 
    # but we can build them efficiently.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # U row: i * m + j
                # V row: j * p + k
                # W row: k * n + i
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Check time
    if time.time() > deadline:
        # Should not happen for this size, but safety guard
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
