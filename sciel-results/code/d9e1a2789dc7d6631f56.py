import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    rank = 48

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_trimmed(i0, j0, k0, keep_col0=True, keep_col1=False):
        """Strassen 2x2x2 with optional zeroing of output columns."""
        nonlocal col
        if col >= rank:
            return False

        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)

        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)

        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        if keep_col0:
            W[c00][col] += 1
        if keep_col1:
            W[c11][col] += 1
        col += 1

        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        if keep_col0:
            W[c10][col] += 1
        if keep_col1:
            W[c11][col] -= 1
        col += 1

        # M3: A00(B11-B00) -> C01 += M3, C11 += M3
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        if keep_col1:
            W[c01][col] += 1
            W[c11][col] += 1
        col += 1

        # M4: A11(B00-A00) -> C00 += M4, C10 += M4
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        if keep_col0:
            W[c00][col] += 1
            W[c10][col] += 1
        col += 1

        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        if keep_col0:
            W[c00][col] -= 1
        if keep_col1:
            W[c01][col] += 1
        col += 1

        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        if keep_col1:
            W[c11][col] += 1
        col += 1

        # M7: (A01-A10)(B10+B11) -> C00 += M7
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        if keep_col0:
            W[c00][col] += 1
        col += 1

        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False

        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)

        e = j0 * p + k0
        g = (j0 + 1) * p + k0

        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1

        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1

        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1

        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1

        return True

    # For each j-block of size 2, use Strassen for k-pairs (0,1),(2,3),(4,5)
    # and a trimmed Strassen for k=6 (only keep col0)
    for j0 in [0, 2]:
        if time.time() > deadline:
            break
        for k0 in [0, 2, 4]:
            if not add_strassen_trimmed(0, j0, k0, keep_col0=True, keep_col1=True):
                break
        # For k=6, only need output column k=6 (col0), not k=7
        if not add_strassen_trimmed(0, j0, 6, keep_col0=True, keep_col1=False):
            pass

    return {"U": U, "V": V, "W": W}
