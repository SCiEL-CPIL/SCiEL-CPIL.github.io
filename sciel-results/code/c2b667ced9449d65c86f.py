import numpy as np

def construct():
    n, m, p = 2, 3, 3
    # Try to find a rank 17 solution?
    # For now, return rank 18.
    # To improve, I need a valid rank 17.
    # I will return rank 18.
    rank = 18
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
