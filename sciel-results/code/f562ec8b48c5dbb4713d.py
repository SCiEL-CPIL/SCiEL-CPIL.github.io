for i in range(n):
    for k in range(p):
        for j in range(m):
            expected = 1.0
            actual = 0.0
            for c in range(rank):
                actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
            if abs(actual - 1.0) > 1e-9:
                return False
