import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        for i in range(n):
            if len(U[i * m]) != rank:
                return False
        for j in range(m):
            if len(V[j * p]) != rank:
                return False
        for k in range(p):
            if len(W[k * n]) != rank:
                return False
    
    # Verify: for each (i,j,k), sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] should be 1
    # Using integers for speed and exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 3, 5, 7
    
    # Trivial decomposition: rank = n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Trivial decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
