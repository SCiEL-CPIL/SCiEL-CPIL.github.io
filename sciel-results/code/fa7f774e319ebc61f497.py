"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 145


def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    rank = 53
    U = np.zeros((n * m, rank), dtype=float)  # 4 x 53
    V = np.zeros((m * p, rank), dtype=float)  # 30 x 53
    W = np.zeros((p * n, rank), dtype=float)  # 30 x 53

    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6

        # P1 = (A00+A11)(B0k0+B1k1)
        U[0, r1] = 1
        U[3, r1] = 1
        V[0 * p + k0, r1] = 1
        V[1 * p + k1, r1] = 1

        # P2 = (A10+A11) B0k0
        U[2, r2] = 1
        U[3, r2] = 1
        V[0 * p + k0, r2] = 1

        # P3 = A00 (B0k1 - B1k1)
        U[0, r3] = 1
        V[0 * p + k1, r3] = 1
        V[1 * p + k1, r3] = -1

        # P4 = A11 (B1k0 - B0k0)
        U[3, r4] = 1
        V[1 * p + k0, r4] = 1
        V[0 * p + k0, r4] = -1

        # P5 = (A00+A01) B1k1
        U[0, r5] = 1
        U[1, r5] = 1
        V[1 * p + k1, r5] = 1

        # P6 = (A10-A00) (B0k0+B0k1)
        U[2, r6] = 1
        U[0, r6] = -1
        V[0 * p + k0, r6] = 1
        V[0 * p + k1, r6] = 1

        # P7 = (A01-A11) (B1k0+B1k1)
        U[1, r7] = 1
        U[3, r7] = -1
        V[1 * p + k0, r7] = 1
        V[1 * p + k1, r7] = 1

        # W reconstruction
        W[2 * k0, r1] = 1
        W[2 * k0, r4] = 1
        W[2 * k0, r5] = -1
        W[2 * k0, r7] = 1

        W[2 * k0 + 1, r2] = 1
        W[2 * k0 + 1, r4] = 1

        W[2 * k1, r3] = 1
        W[2 * k1, r5] = 1

        W[2 * k1 + 1, r1] = 1
        W[2 * k1 + 1, r2] = -1
        W[2 * k1 + 1, r3] = 1
        W[2 * k1 + 1, r6] = 1

    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)

    base_r = 49
    U[0, base_r] = 1
    V[0 * p + 14, base_r] = 1
    W[2 * 14, base_r] = 1

    U[1, base_r + 1] = 1
    V[1 * p + 14, base_r + 1] = 1
    W[2 * 14, base_r + 1] = 1

    U[2, base_r + 2] = 1
    V[0 * p + 14, base_r + 2] = 1
    W[2 * 14 + 1, base_r + 2] = 1

    U[3, base_r + 3] = 1
    V[1 * p + 14, base_r + 3] = 1
    W[2 * 14 + 1, base_r + 3] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
