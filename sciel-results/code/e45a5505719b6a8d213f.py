"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave margin for evaluator overhead

    n, m, p = DIMENSIONS

    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # At this point we have a valid rank-125 decomposition.
    # We could attempt to find a lower rank decomposition within the remaining time,
    # but without known explicit low-rank tensors for 5x5x5 and the high risk of
    # failure/timeout, we return the guaranteed valid schoolbook result.
    # (Any optimization here must be verified to be exact, which is non-trivial for 5^3.)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
