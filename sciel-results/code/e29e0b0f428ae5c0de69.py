"""Matrix multiplication 4x4 by 4x5 via hybrid Strassen + bilinear column block."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5

    # Canonical Strassen M terms: each M = sum of (a_block, b_block, sign).
    # a_block=(ar,ac) A-block, b_block=(br,bc) B-block (B is square 4x4 here).
    Mterms = [
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],   # M1=(A00+A11)(B00+B11)
        [(0, 1, 1), (0, 0, -1)],                          # M2=A00(B11-B10)
        [(1, 0, 1), (1, 1, 1), (0, 0, -1), (0, 1, -1)],  # M3=(A10-A00)(B00+B01)
        [(1, 1, 1), (1, 0, -1)],                          # M4=A11(B01-B00)
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],   # M5=(A00+A01)(B10+B11)
        [(1, 0, 1), (1, 1, -1), (0, 0, 1), (0, 1, -1)],  # M6=(A10+A11)(B00-B01)
        [(0, 0, 1), (0, 1, 1), (1, 0, -1), (1, 1, -1)],  # M7=(A01-A11)(B10+B11)
    ]
    Cdefs = [
        [(1, 1), (4, 1), (5, -1), (7, -1)],   # C00
        [(2, 1), (5, 1)],                       # C01
        [(3, 1), (6, 1)],                       # C10
        [(1, 1), (2, -1), (3, 1), (4, -1)],    # C11
    ]

    def build():
        rank = 7 * 4 + 4 * 4  # 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            ri = i // 2
            colbase = i * 7
            for mi in range(7):
                c = colbase + mi
                for (ar, ac, sign) in Mterms[mi]:
                    for lj in range(2):
                        U[i * 4 + ac * 2 + lj][c] += sign
                        V[ar * 2 * 4 + ac * 2 + lj][c] += sign
                for (mi2, coef) in Cdefs[ri]:
                    for cj in range(2):
                        for lj in range(2):
                            W[(cj * 2 + lj) * n + i][c] += coef
        for i in range(n):
            colbase = 28 + i * 4
            for j in range(m):
                c = colbase + j
                U[i * 4 + j][c] = 1
                V[j * 4 + 4][c] = 1
                W[4 * n + i][c] = 1
        return U, V, W, rank

    def verify(U, V, W, rank):
        rng = np.random.default_rng(999)
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            Ctrue = A @ B
            C = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i, r] * V[k, r] * W[k * n + i, r]
                    C[i, k] = s
            if not np.allclose(C, Ctrue, atol=1e-6):
                return False
        return True

    def to_int_list(M):
        M = np.round(M * 2) / 2
        return [[float(M[r, c]) for c in range(M.shape[1])] for r in range(M.shape[0])]

    def schoolbook():
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

    U, V, W, rank = build()
    if verify(U, V, W, rank):
        return {"U": to_int_list(U), "V": to_int_list(V), "W": to_int_list(W)}
    return schoolbook()
# EVOLVE-BLOCK-END
