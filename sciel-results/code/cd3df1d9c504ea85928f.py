for i in range(n):
    for k in range(p):
        U[i * m][col] = 1
        V[k][col] = 1
        W[k * n + i][col] = 1
        col += 1
