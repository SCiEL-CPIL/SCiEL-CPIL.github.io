    for k in range(6):
        col = col_off + k
        for i in range(2):
            U[i * m + 0, col] += 1.0
        V[0 * p + k, col] += 1.0
        for i in range(2):
            W[k * n + i, col] += 1.0
