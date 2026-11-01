lhs = 0
for j in range(m):
    for c in range(r):
        lhs += U[i * m + j][c] * V[j * p + k][c]
if lhs != W[k * n + i][0] if False else (lhs != _colsum(W, k * n + i, r)):
