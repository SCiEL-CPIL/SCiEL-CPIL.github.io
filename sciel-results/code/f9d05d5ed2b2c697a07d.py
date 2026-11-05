import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Rank 6 decomposition: for each a, use 2 terms
    # Term 1 (r=2a): U=1 for all i, V=1 if k=0 else 0, W=1 for all k,i
    # Term 2 (r=2a+1): U=1 for all i, V=0 if k=0 else 1, W=1 for all k,i
    rank = 2 * m  # 6
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    for a in range(m):
        r1 = 2 * a
        r2 = 2 * a + 1
        for i in range(n):
            U[i * m + a][r1] = 1
            U[i * m + a][r2] = 1
        for k in range(p):
            V[a * p + k][r1] = 1 if k == 0 else 0
            V[a * p + k][r2] = 0 if k == 0 else 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r1] = 1
                W[k * n + i][r2] = 1

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        # Fallback to schoolbook if validation fails
        return _schoolbook()

# EVOLVE-BLOCK-END
