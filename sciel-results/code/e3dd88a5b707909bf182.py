import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0

    n, m, p = DIMENSIONS  # 6, 6, 6

    # Standard schoolbook decomposition
    # Rank is n * m * p
    rank = n * m * p

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    rank_idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U corresponds to A[i, k]
                # Index in U: i * m + k
                U[i * m + k][rank_idx] = 1

                # V corresponds to B[k, j]
                # Index in V: k * p + j
                V[k * p + j][rank_idx] = 1

                # W corresponds to C[j, i]
                # Index in W: j * n + i
                W[j * n + i][rank_idx] = 1

                rank_idx += 1

                if time.time() > deadline - 5.0:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
