"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    best_rank = n * m * p
    best_U = [[0.0] * best_rank for _ in range(n * m)]
    best_V = [[0.0] * best_rank for _ in range(m * p)]
    best_W = [[0.0] * best_rank for _ in range(p * n)]
    best_score = 1e10

    random.seed(12345)

    for trial in range(500):
        if time.time() > deadline - 5:
            break
        if trial == 0:
            rank = best_rank
        else:
            rank = 42 + random.randint(0, 5)

        U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]

        for i in range(3000):
            if time.time() > deadline - 5:
                break
            # Compute gradient
            gU = [[0.0] * rank for _ in range(n * m)]
            gV = [[0.0] * rank for _ in range(m * p)]
            gW = [[0.0] * rank for _ in range(p * n)]
            score = 0.0

            for c in range(rank):
                uc = [U[r][c] for r in range(n * m)]
                vc = [V[r][c] for r in range(m * p)]
                wc = [W[r][c] for r in range(p * n)]

                for a in range(n):
                    for b in range(m):
                        ua = uc[a * m + b]
                        for d in range(p):
                            term = ua * vc[b * p + d] * wc[d * n + a]
                            score += term * term
                            gU[a * m + b][c] += 2.0 * term * vc[b * p + d] * wc[d * n + a]
                            gV[b * p + d][c] += 2.0 * term * ua * wc[d * n + a]
                            gW[d * n + a][c] += 2.0 * term * ua * vc[b * p + d]

            # SGD update
            lr = 0.001
            for r in range(n * m):
                for c in range(rank):
                    U[r][c] -= lr * gU[r][c]
            for r in range(m * p):
                for c in range(rank):
                    V[r][c] -= lr * gV[r][c]
            for r in range(p * n):
                for c in range(rank):
                    W[r][c] -= lr * gW[r][c]

            if score < 1e-4:
                best_score = score
                best_rank = rank
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
                break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
