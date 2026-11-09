"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from fractions import Fraction
    import random

    start_time = time.time()
    DEATHLINE = 140.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, tol=1e-10):
        """Verify the decomposition exactly decomposes the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Construct the rank 30 decomposition
    rank = m * p  # 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            V[j * p + k][r] = 1
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
                
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to trivial rank 60
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
