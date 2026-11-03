"""Matrix multiplication 2x3x10 tensor decomposition."""
DIMENSIONS = [2, 3, 10]

def construct():
    """
    Schoolbook decomposition of rank n*m*p for multiplying (n x m) by (m x p).
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W uses trace convention: W[k*n + i, c]
    """
    n, m, p = DIMENSIONS
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
