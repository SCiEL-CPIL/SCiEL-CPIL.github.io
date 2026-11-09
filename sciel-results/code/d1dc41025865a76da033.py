"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Use numpy for efficient storage and potential future optimization
    import numpy as np
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # The task requires integer/half-integer entries. 1.0 is an integer.
    # We return the numpy arrays.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
