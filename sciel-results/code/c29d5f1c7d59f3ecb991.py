"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 2, 5, 6

    # Target Rank 54 based on Strassen decomposition of 2x2 blocks
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def _add_strassen_2x2(i0, j0, k0):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1

        # M1: (A10+A11)B00
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + (i0 + 1)][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] -= 1
        col += 1

        # M2: A00(B01-B11)
        U[i0 * m + j0][col] += 1
        V[j0 * p + (k0 + 1)][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1

        # M3: A11(B10-B00)
        U[(i0 + 1) * m + (j0 + 1)][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0 + 1)][col] += 1
        col += 1

        # M4: (A00+A01)B11
        U[i0 * m + j0][col] += 1
        U[i0 * m + (j0 + 1)][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        col += 1

        # M5: (A10-A00)(B00+B01)
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + (k0 + 1)][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1

        # M6: (A01-A11)(B10+B11)
        U[i0 * m + (j0 + 1)][col] += 1
        U[(i0 + 1) * m + (j0 + 1)][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + (k0 + 1)][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

    def _add_direct_2x1(i0, j, k0):
        nonlocal col
        U[i0 * m + j][col] += 1
        V[j * p + k0][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

        U[(i0 + 1) * m + j][col] += 1
        V[j * p + (k0 + 1)][col] += 1
        W[(k0 + 1) * n + (i0 + 1)][col] += 1
        col += 1

    # j=0,1: Strassen 2x2x2 for k=0,1
    _add_strassen_2x2(0, 0, 0)
    # j=0,1: Strassen 2x2x2 for k=2,3
    _add_strassen_2x2(0, 0, 2)
    # j=0,1: Strassen 2x2x2 for k=4,5
    _add_strassen_2x2(0, 0, 4)

    # j=2,3: Strassen 2x2x2 for k=0,1
    _add_strassen_2x2(0, 2, 0)
    # j=2,3: Strassen 2x2x2 for k=2,3
    _add_strassen_2x2(0, 2, 2)
    # j=2,3: Strassen 2x2x2 for k=4,5
    _add_strassen_2x2(0, 2, 4)

    # j=4: direct 2x1x2 for k=0,1
    _add_direct_2x1(0, 4, 0)
    # j=4: direct 2x1x2 for k=2,3
    _add_direct_2x1(0, 4, 2)
    # j=4: direct 2x1x2 for k=4,5
    _add_direct_2x1(0, 4, 4)

    if col != 54:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
