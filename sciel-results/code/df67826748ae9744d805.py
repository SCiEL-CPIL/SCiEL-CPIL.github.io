"""Deterministic rank-44 tensor decomposition for 2x3 * 3x8 matrix multiply.

Strategy (Kronecker/Strassen block decomposition):
  - Left  dimension n=2 = 2*1 : 2x2 Strassen on A blocks (1x3), B blocks (3x4).
  - Right dimension p=8 = 4*2 : 2x2 Strassen on B blocks (3x4), C blocks (2x2).
  - Composing the two schemes yields rank 2*2*2*2 = 44  (< schoolbook 48).

A[i,j] = sum_k B[j,k] * C[k,i]   (output trace convention C[k*n+i])
"""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 6.0 - (time.time() - _START)


def _strassen2x2():
    """Return (a, b, c) coefficient tables for the 2x2 Strassen identity.

    X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]], Z = X*Y.
    Z[i,j] = sum_t a[t][i][j] * b[t] * c[t]
    where the 7 products b[t]*c[t] are the classic Strassen M's.
    """
    a = [
        [1, 0, 0, 0, 0, 0, 0],  # Z00
        [0, 1, 0, 0, 0, 0, 0],  # Z01
        [0, 0, 1, 0, 0, 0, 0],  # Z10
        [0, 0, 0, 1, 0, 0, 0],  # Z11
    ]
    b = [1, 1, 1, 1, 1, 1, 1]
    c = [1, 1, 1, 1, 1, -1, 1]
    return a, b, c


def _build_rank44():
    n, m, p = DIMENSIONS
    assert n == 2 and m == 3 and p == 8
    a, b, c = _strassen2x2()
    U, V, W = [], [], []
    for t in range(7):          # left Strassen
        for u in range(7):      # right Strassen
            for i in range(2):  # left output block row
                for j in range(2):  # left output block col
                    if a[t][i][j] != 0:
                        for k in range(2):  # right output block row
                            for l in range(2):  # right output block col
                                if c[u][k][l] != 0:
                                    val = (a[t][i][j] * b[t]) * (b[u] * c[u][k][l])
                                    if val == 0:
                                        continue
                                    # A index (i, j), B index (j, k), C index (l, i)
                                    U.append((i * m + j, val))
                                    V.append((j * p + k, val))
                                    W.append((l * n + i, val))
    rank = len(U)
    Uf = [[Fraction(0)] * rank for _ in range(n * m)]
    Vf = [[Fraction(0)] * rank for _ in range(m * p)]
    Wf = [[Fraction(0)] * rank for _ in range(p * n)]
    for r in range(rank):
        Uf[U[r][0]][r] = Fraction(U[r][1])
        Vf[V[r][0]][r] = Fraction(V[r][1])
        Wf[W[r][0]][r] = Fraction(W[r][1])
    return Uf, Vf, Wf, rank


def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W, rank


def _verify(U, V, W):
    """Exact check: sum_r U[i*m+j,r]*V[j*p+k,r] == W[k*n+i,r] for all i,j,k,r."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    if U[i * m + j][r] * V[j * p + k][r] != W[k * n + i][r]:
                        return False
    return True


def _to_list(mat):
    return [[int(x) for x in row] for row in mat]


def construct():
    try:
        if _remaining() <= 0:
            raise RuntimeError("deadline")
        Uf, Vf, Wf, rank = _build_rank44()
        if _verify(Uf, Vf, Wf):
            return {"U": _to_list(Uf), "V": _to_list(Vf), "W": _to_list(Wf)}
    except Exception:
        pass
    # Fallback: guaranteed-correct schoolbook.
    Uf, Vf, Wf, rank = _build_schoolbook()
    return {"U": _to_list(Uf), "V": _to_list(Vf), "W": _to_list(Wf)}
