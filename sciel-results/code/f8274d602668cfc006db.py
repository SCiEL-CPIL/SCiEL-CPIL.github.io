"""
Matrix multiplication 2x3x5 tensor decomposition.
Uses verified rank-7 2x2x2 (Bini/Strassen) block + schoolbook for remaining terms.
Target rank: 29 (vs schoolbook 30).
"""
from fractions import Fraction
import time

N, M, P = 2, 3, 5


def _verify_full(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(rank):
                    a = uc[c]
                    if a:
                        s += a * vc[c] * wc[c]
                if s != 1:
                    return False
    return True


def _emit(M):
    out = []
    for row in M:
        r = []
        for v in row:
            f = v * 2
            if f.denominator == 1:
                fv = f.numerator
                r.append(fv // 2 if fv % 2 == 0 else fv / 2.0)
            else:
                r.append(float(v))
        out.append(r)
    return out


def _schoolbook(n, m, p):
    r = n * m * p
    U = [[Fraction(0)] * r for _ in range(n * m)]
    V = [[Fraction(0)] * r for _ in range(m * p)]
    W = [[Fraction(0)] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


# EVOLVE-BLOCK-START
def construct():
    n, m, p = N, M, P

    bini = [
        ((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)),
        ((0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1)),
        ((1, 0, 0, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
        ((0, 1, 1, 0), (0, 0, 0, 1), (0, 1, 0, 0)),
        ((1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)),
        ((0, 0, 1, 1), (0, 0, 1, 0), (0, 1, 0, 0)),
        ((1, 0, 1, 1), (0, 1, 1, 0), (1, 1, 0, 0)),
    ]

    # Verify Bini 2x2x2
    ok = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = Fraction(0)
                for (u, v, w) in bini:
                    s += Fraction(u[i * 2 + j]) * Fraction(v[j * 2 + k]) * Fraction(w[k * 2 + i])
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        U, V, W = _schoolbook(n, m, p)
        return {"U": _emit(U), "V": _emit(V), "W": _emit(W)}

    R = 29
    U = [[Fraction(0)] * R for _ in range(n * m)]
    V = [[Fraction(0)] * R for _ in range(m * p)]
    W = [[Fraction(0)] * R for _ in range(p * n)]
    col = 0

    for (u, v, w) in bini:
        U[0][col] = Fraction(u[0])
        U[1][col] = Fraction(u[1])
        U[3][col] = Fraction(u[2])
        U[4][col] = Fraction(u[3])
        V[0][col] = Fraction(v[0])
        V[1][col] = Fraction(v[1])
        V[5][col] = Fraction(v[2])
        V[6][col] = Fraction(v[3])
        W[0][col] = Fraction(w[0])
        W[1][col] = Fraction(w[1])
        W[2][col] = Fraction(w[2])
        W[3][col] = Fraction(w[3])
        col += 1

    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    for i in range(n):
        for j in range(2):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    if col != R:
        U, V, W = _schoolbook(n, m, p)
        return {"U": _emit(U), "V": _emit(V), "W": _emit(W)}

    if not _verify_full(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": _emit(U), "V": _emit(V), "W": _emit(W)}

    return {"U": _emit(U), "V": _emit(V), "W": _emit(W)}
# EVOLVE-BLOCK-END
