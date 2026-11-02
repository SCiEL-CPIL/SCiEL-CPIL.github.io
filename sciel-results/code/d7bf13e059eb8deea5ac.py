"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s / 150s limit
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's rank-7 algorithm applied independently to each of the 5
    # disjoint column-pairs of the 2x10 matrix. The 2x2 A-factors are shared
    # across blocks; the B/C supports are disjoint, so the tensor rank is the
    # sum of the per-block ranks = 5 * 7 = 35 (optimal for this tensor).
    for block in range(5):
        if time.time() > deadline:
            break
        k1 = 2 * block
        k2 = 2 * block + 1

        # M1: (A00+A11)(B00+B11)
        r = 7 * block + 0
        U[0][r] += 1
        U[3][r] += 1
        V[k1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] += 1
        W[2 * k2 + 1][r] += 1

        # M2: (A10+A11)B00
        r = 7 * block + 1
        U[2][r] += 1
        U[3][r] += 1
        V[k1][r] += 1
        W[2 * k1 + 1][r] += 1
        W[2 * k2 + 1][r] -= 1

        # M3: A00(B01-B11)
        r = 7 * block + 2
        U[0][r] += 1
        V[k2][r] += 1
        V[10 + k2][r] -= 1
        W[2 * k2][r] += 1
        W[2 * k2 + 1][r] += 1

        # M4: A11(B10-B00)
        r = 7 * block + 3
        U[3][r] += 1
        V[10 + k1][r] += 1
        V[k1][r] -= 1
        W[2 * k1][r] += 1
        W[2 * k1 + 1][r] += 1

        # M5: (A00+A01)B11
        r = 7 * block + 4
        U[0][r] += 1
        U[1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] -= 1
        W[2 * k2][r] += 1

        # M6: (A10-A00)(B00+B01)
        r = 7 * block + 5
        U[2][r] += 1
        U[0][r] -= 1
        V[k1][r] += 1
        V[k2][r] += 1
        W[2 * k2 + 1][r] += 1

        # M7: (A01-A11)(B10+B11)
        r = 7 * block + 6
        U[1][r] += 1
        U[3][r] -= 1
        V[10 + k1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
