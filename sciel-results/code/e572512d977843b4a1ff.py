import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 145

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 4*5*5 = 100
    # For 4x5x5, no simple rank reduction is known that is easily implementable
    # and guaranteed to be correct within the time limit without extensive search.
    # We return the schoolbook decomposition.
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

# EVOLVE-BLOCK-END
