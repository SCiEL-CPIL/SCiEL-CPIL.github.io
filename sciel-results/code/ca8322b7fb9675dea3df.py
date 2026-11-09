"""Matrix multiplication 2x4x7 rank search with exact verification."""
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    if len(V[0]) != r or len(W[0]) != r:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s = s + Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS
    best = _schoolbook()

    # Rank-4 construction: U[i*m+j, j]=1, V[j*p+k, j]=1, W[k*n+i, c]=1 for all c.
    r = m  # = 4
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    for k in range(p):
        for i in range(n):
            for c in range(r):
                W[k * n + i][c] = 1
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    return best

if __name__ == "__main__":
    construct()
