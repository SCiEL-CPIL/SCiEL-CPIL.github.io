for i in range(n):
    for j in range(m):
        for k in range(p):
            c = i * m * p + j * p + k
            U[i * m + j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n + i][c] = 1.0
