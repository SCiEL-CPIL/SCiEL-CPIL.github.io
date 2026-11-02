import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r corresponds to (a, b) where a is row index of A/C, b is col index of A / row index of B
    r = 0
    for a in range(n):
        for b in range(m):
            # U index: i * m + k. Here i=a, k=b.
            U[a * m + b][r] = 1
            # V index: k * p + j. Here k=b. j can be anything, but we set it for all j?
            # No, V is a matrix. We need V[k * p + j, r] to be 1 for all j?
            # In the derivation, V_{k', j, r} = delta_{k', b}.
            # So for a fixed r (a,b), V[k * p + j, r] = 1 if k == b, else 0.
            for j in range(p):
                V[b * p + j][r] = 1
            # W index: j * n + i. Here i=a. j can be anything.
            # W_{j, i', r} = delta_{i', a}.
            # So for a fixed r (a,b), W[j * n + i, r] = 1 if i == a, else 0.
            for j in range(p):
                W[j * n + a][r] = 1
            r += 1
            
    return {"U": U, "V": V, "W": W}
