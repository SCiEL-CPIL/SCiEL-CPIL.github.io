import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = 150.0
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 5.0:
                    break
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
            
    # If we broke early due to time (highly unlikely), we return partial?
    # No, partial is invalid. But 96 iterations is < 1ms.
    # So we just return the full result.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
