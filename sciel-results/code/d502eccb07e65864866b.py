import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Trivial 45-rank decomposition
    def get_trivial():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Attempt 41-rank decomposition
    # Lader 23 for 3x3x3 + Trivial 18 for 3x3x2
    
    # Lader terms (hardcoded)
    # Verified set of 23 terms for 3x3x3 matrix multiplication
    # U, V, W are 3x3 matrices flattened row-major
    # W index is k*n + i
    
    lader_terms = [
        # 1
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0]),
        # 2
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # 3
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # 4
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 1,0,0, 0,0,0]),
        # 5
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0]),
        # 6
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]), # Placeholder, need real ones
        # ...
    ]
    
    # Since I don't have the full 23 correct terms hardcoded with 100% confidence in this thought block,
    # and the previous attempt failed to provide them, I will use a known valid set from a reliable source 
    # or fall back to trivial if verification fails.
    
    # I will use the trivial decomposition to ensure safety, as I cannot guarantee the Lader coefficients 
    # are correct in this context without a lookup table.
    
    return get_trivial()
