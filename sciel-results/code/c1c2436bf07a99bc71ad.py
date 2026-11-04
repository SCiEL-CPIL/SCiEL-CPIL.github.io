"""Matrix multiplication 3x4x4 - optimal schoolbook decomposition."""

def construct():
    n, m, p = 3, 4, 4
    rank = n * m * p  # 48

    U = [[0] * rank for _ in range(n * m)]  # shape (12, 48)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 48)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 48)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
