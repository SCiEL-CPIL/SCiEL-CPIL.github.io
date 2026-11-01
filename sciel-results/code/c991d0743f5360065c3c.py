def _build_rank_8(n, m, p):
    total_rank = 8
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for s in range(total_rank):
        if s < 4:
            j = s
            i = 0
            U[i * m + j][s] = 1
            for k in range(p):
                V[j * p + k][s] = 1
            for k in range(p):
                W[k * n + i][s] = 1
        else:
            j = s - 4
            i = 1
            U[i * m + j][s] = 1
            for k in range(p):
                V[j * p + k][s] = 1
            for k in range(p):
                W[k * n + i][s] = 1

    return U, V, W
