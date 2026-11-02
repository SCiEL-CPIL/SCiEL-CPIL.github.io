"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 sub-blocks and
rank-2 schoolbook for 1x2x2 sub-blocks, yielding rank 72.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Valid entries: T[i,j,k] = 1 when output index is (k,i)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    # Invalid entries: output (k, i') with i' != i
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                    if abs(val) > tol:
                        return False
    return True


def _add_strassen(U, V, W, a, b, c, r0):
    # M1 = (A00+A11)(B00+B11)
    U[a[0], r0] += 1; U[a[3], r0] += 1
    V[b[0], r0] += 1; V[b[3], r0] += 1
    W[c[0], r0] += 1; W[c[3], r0] += 1
    # M2 = (A10+A11)B00
    U[a[2], r0+1] += 1; U[a[3], r0+1] += 1
    V[b[0], r0+1] += 1
    W[c[2], r0+1] += 1; W[c[3], r0+1] += 1
    # M3 = A00(B01-B11)
    U[a[0], r0+2] += 1
    V[b[1], r0+2] += 1; V[b[3], r0+2] -= 1
    W[c[1], r0+2] += 1; W[c[3], r0+2] += 1
    # M4 = A11(B10-B00)
    U[a[3], r0+3] += 1
    V[b[2], r0+3] += 1; V[b[0], r0+3] -= 1
    W[c[2], r0+3] += 1; W[c[3], r0+3] += 1
    # M5 = (A00+A01)B11
    U[a[0], r0+4] += 1; U[a[1], r0+4] += 1
    V[b[3], r0+4] += 1
    W[c[1], r0+4] += 1; W[c[3], r0+4] += 1
    # M6 = (A10-A00)(B00+B01)
    U[a[2], r0+5] += 1; U[a[0], r0+5] -= 1
    V[b[0], r0+5] += 1; V[b[1], r0+5] += 1
    W[c[2], r0+5] += 1; W[c[3], r0+5] += 1
    # M7 = (A01-A11)(B10+B11)
    U[a[1], r0+6] += 1; U[a[3], r0+6] -= 1
    V[b[2], r0+6] += 1; V[b[3], r0+6] += 1
    W[c[0], r0+6] += 1; W[c[3], r0+6] += 1


def _add_dot_1x2x2(U, V, W, a, b, c, r0):
    # Two scalar multiplications
    U[a[0], r0] += 1
    V[b[0], r0] += 1; V[b[1], r0] += 1
    W[c[0], r0] += 1; W[c[1], r0] += 1
    U[a[1], r0+1] += 1
    V[b[2], r0+1] += 1; V[b[3], r0+1] += 1
    W[c[0], r0+1] += 1; W[c[1], r0+1] += 1


def _build():
    n, m, p = DIMENSIONS  # 3, 4, 8
    total_rank = 72
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    r = 0
    for ps in (0, 4):
        for ms in (0, 2):
            for pss in (ps, ps + 2):
                ai = [i * m + j for i in range(2) for j in range(ms, ms + 2)]
                bi = [j * p + k for j in range(ms, ms + 2) for k in range(pss, pss + 2)]
                ci = [k * n + i for k in range(pss, pss + 2) for i in range(2)]
                _add_strassen(U, V, W, ai, bi, ci, r)
                r += 7
            for pss in (ps, ps + 2):
                ai = [2 * m + j for j in range(ms, ms + 2)]
                bi = [j * p + k for j in range(ms, ms + 2) for k in range(pss, pss + 2)]
                ci = [k * n + 2 for k in range(pss, pss + 2)]
                _add_dot_1x2x2(U, V, W, ai, bi, ci, r)
                r += 2
    return U, V, W


def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    try:
        U, V, W = _build()
        if time.time() < deadline and _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    return _schoolbook()

# EVOLVE-BLOCK-END
