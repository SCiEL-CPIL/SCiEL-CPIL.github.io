for io in range(p * n):
    for u in range(n * m):
        for v in range(m * p):
            s = 0
            for c in range(rank):
                s += U[u][c] * V[v][c] * W[io][c]
