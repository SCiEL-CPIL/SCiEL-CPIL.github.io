for k in range(p):  # k from 0 to 4
    for j in range(m):  # j from 0 to 4
        for i in range(n):  # i from 0 to 3
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
