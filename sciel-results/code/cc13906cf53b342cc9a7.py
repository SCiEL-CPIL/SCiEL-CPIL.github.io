"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a bilinear decomposition for (2,5,5) matrix multiplication.
    We use the schoolbook decomposition with rank n*m*p = 50.
    """
    import itertools
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n

    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
