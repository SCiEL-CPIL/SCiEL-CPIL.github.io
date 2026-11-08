val = 0.0
for r in range(rank):
    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
if abs(val - 1.0) > 1e-9:
    return False
