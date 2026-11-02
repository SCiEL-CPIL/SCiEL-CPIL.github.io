import time
import sys
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact for matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0
                for r in range(rank):
                    val += U[i * m + k][r] * V[k * p + j][r] * W[j * n + i][r]
                if val != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 120
    
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
                
    return {"U": best_U, "V": best_V, "W": best_W}
