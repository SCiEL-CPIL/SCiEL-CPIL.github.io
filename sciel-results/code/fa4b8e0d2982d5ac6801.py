import time
import numpy as np

def construct():
    """
    Rank-42 exact tensor decomposition for 2x2 times 2x12 matrix multiplication.
    Uses Strassen's rank-7 algorithm on 6 independent column pairs.
    """
    n, m, p = 2, 2, 12
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42

    U = np.zeros((n * m, rank), dtype=int)  # (4, 42)
    V = np.zeros((m * p, rank), dtype=int)  # (24, 42)
    W = np.zeros((p * n, rank), dtype=int)  # (24, 42)

    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
                U[0, c] = 1; U[3, c] = 1
                V[0 * p + 2*t, c] = 1; V[0 * p + 2*t+1, c] = 1
                V[1 * p + 2*t, c] = 1; V[1 * p + 2*t+1, c] = 1
                W[(2*t) * n + 0, c] = 1   # C00
                W[(2*t+1) * n + 1, c] = 1 # C11
            elif s == 1:
                # M2 = (A10+A11)B00 -> contributes to C10 (+1) and C11 (-1)
                U[2, c] = 1; U[3, c] = 1
                V[0 * p + 2*t, c] = 1
                W[(2*t) * n + 1, c] = 1   # C10
                W[(2*t+1) * n + 1, c] = -1 # C11
            elif s == 2:
                # M3 = A00(B10-B11) -> contributes to C01 (+1) and C11 (+1)
                U[0, c] = 1
                V[1 * p + 2*t, c] = 1; V[1 * p + 2*t+1, c] = -1
                W[(2*t) * n + 1, c] = 1   # C01
                W[(2*t+1) * n + 1, c] = 1 # C11
            elif s == 3:
                # M4 = A11(B01-B00) -> contributes to C00 (+1) and C10 (+1)
                U[3, c] = 1
                V[0 * p + 2*t+1, c] = 1; V[0 * p + 2*t, c] = -1
                W[(2*t) * n + 0, c] = 1   # C00
                W[(2*t) * n + 1, c] = 1   # C10
            elif s == 4:
                # M5 = (A00+A01)B11 -> contributes to C00 (-1) and C01 (+1)
                U[0, c] = 1; U[1, c] = 1
                V[1 * p + 2*t+1, c] = 1
                W[(2*t) * n + 0, c] = -1  # C00
                W[(2*t) * n + 1, c] = 1   # C01
            elif s == 5:
                # M6 = (A10-A00)(B00+B01) -> contributes to C11 (+1)
                U[2, c] = 1; U[0, c] = -1
                V[0 * p + 2*t, c] = 1; V[0 * p + 2*t+1, c] = 1
                W[(2*t+1) * n + 1, c] = 1 # C11
            elif s == 6:
                # M7 = (A01-A11)(B10+B11) -> contributes to C00 (+1)
                U[1, c] = 1; U[3, c] = -1
                V[1 * p + 2*t, c] = 1; V[1 * p + 2*t+1, c] = 1
                W[(2*t) * n + 0, c] = 1   # C00

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
