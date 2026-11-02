"""Program for 3x3 matrix multiplication tensor decomposition."""

def construct():
    n = m = p = 3
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for l in range(m):
            for j in range(p):
                U[i * m + l][column] = 1
                V[l * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
