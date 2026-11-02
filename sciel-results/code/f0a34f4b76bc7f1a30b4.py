import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Trivial Kronecker-product decomposition: rank = n*m*p
    rank = n * m * p

    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}
