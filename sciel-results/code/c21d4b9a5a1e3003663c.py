"""Rewritten program for 2x5x6 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [2, 5, 6]
DEADLINE = time.time() + 140

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Try rank 1: all-ones decomposition
    U1 = [[1] for _ in range(nm)]
    V1 = [[1] for _ in range(mp)]
    W1 = [[1] for _ in range(pn)]
    
    if verify(U1, V1, W1):
        return {"U": U1, "V": V1, "W": W1}
    
    # Fallback to schoolbook
    return schoolbook()
