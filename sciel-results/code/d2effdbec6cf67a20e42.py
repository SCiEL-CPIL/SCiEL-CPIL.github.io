# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for idx in range(rank):
        k = idx % p
        j = (idx // p) % m
        i = idx // (p * m)
        U[i * m + j][idx] = 1
        V[j * p + k][idx] = 1
        W[k * n + i][idx] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
