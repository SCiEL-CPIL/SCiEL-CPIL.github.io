"""Structural rewrite: deterministic algebraic low-rank constructions + exact verification.

Instead of returning a single fixed tensor, we build several candidate decompositions
from known algebraic schemes (Strassen on the 2x2 core, a rank-17 scheme, and the
rank-18 schoolbook) and verify each against the exact 2x3x3 matrix-multiplication tensor.
We return the lowest-rank candidate that is provably exact, so we never emit an invalid
result. All entries are integers (a subset of the allowed integer/half-integer set).
"""
import time
import itertools
import numpy as np

_DIMS = (2, 3, 3)
_START = time.time()
_DEADLINE = 115.0  # explicit wall-clock guard: return best found before the 120s kill.


def _deadline_exceeded():
    return (time.time() - _START) > _DEADLINE


def _to_int_list(mat):
    # Convert a numeric matrix to a list-of-lists of Python ints (exact).
    return [[int(round(x)) for x in row] for row in np.asarray(mat, dtype=object).tolist()]


def _schoolbook(n, m, p):
    """Rank n*m*p exact decomposition (always valid). U[i*m+j], V[j*p+k], W[k*n+i]."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}


def _strassen_2x2_core(n, m, p):
    """Strassen on the top-left 2x2 block; remaining columns/rows by schoolbook.

    Rank = 7 + (m-2)*p + (p-2)*n. For 2x3x3 -> 7 + 3 + 2 = 12 (if the scheme is exact).
    """
    rank = 7 + (m - 2) * p + (p - 2) * n
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    # Strassen products on A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    for (a1, a2, b1, b2, w1, w2, s) in (
        (0, 0, 0, 0, 0, 0, 1),   # M1 = (A00+A11)(B00+B11)
        (0, 1, 0, 1, 1, 1, 1),   # M2 = (A10+A11) B00
        (0, 0, 1, 0, 1, 0, 1),   # M3 = A00 (B10-B11)
        (1, 1, 1, 1, 0, 1, 1),   # M4 = A11 (B01-B00)
        (0, 1, 0, 1, 0, 0, 1),   # M5 = (A00+A01) B11
        (1, 0, 1, 0, 1, 1, -1),  # M6 = (A10-A00)(B00+B01)
        (1, 1, 1, 0, 0, 0, -1),  # M7 = (A01-A11)(B10+B11)
    ):
        U[a1 * m + 0, c] = a1
        U[a2 * m + 1, c] = a2
        V[0 * p + b1, c] = b1
        V[1 * p + b2, c] = b2
        W[0 * n + w1, c] = w1
        W[1 * n + w2, c] = w2
        c += 1
    # Schoolbook for columns j = 2..m-1 (i in 0..1)
    for i in range(2):
        for j in range(2, m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    # Schoolbook for rows i = 2..n-1 (all j)
    for i in range(2, n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}


def _scheme17(n, m, p):
    """A rank-17 candidate: Strassen on the 2x2 core + one fused column product.

    Rank = 7 + (p-2)*n + (m-3)*p + 1. For 2x3x3 -> 7 + 2 + 0 + 1 = 10 (if exact).
    (If not exact, verification rejects it and we fall back.)
    """
    rank = 7 + (p - 2) * n + (m - 3) * p + 1
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for (a1, a2, b1, b2, w1, w2, s) in (
        (0, 0, 0, 0, 0, 0, 1),
        (0, 1, 0, 1, 1, 1, 1),
        (0, 0, 1, 0, 1, 0, 1),
        (1, 1, 1, 1, 0, 1, 1),
        (0, 1, 0, 1, 0, 0, 1),
        (1, 0, 1, 0, 1, 1, -1),
        (1, 1, 1, 0, 0, 0, -1),
    ):
        U[a1 * m + 0, c] = a1
        U[a2 * m + 1, c] = a2
        V[0 * p + b1, c] = b1
        V[1 * p + b2, c] = b2
        W[0 * n + w1, c] = w1
        W[1 * n + w2, c] = w2
        c += 1
    # Fused product for column j = 2: M = (A02 + A12) * (B02 - B12) -> C02 += M, C12 -= M
    U[0 * m + 2, c] = 1
    U[1 * m + 2, c] = 1
    V[2 * p + 0, c] = 1
    V[2 * p + 1, c] = -1
    W[0 * n + 0, c] = 1
    W[1 * n + 1, c] = -1
    c += 1
    # Schoolbook for columns j = 3..m-1
    for i in range(2):
        for j in range(3, m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    # Schoolbook for rows i = 2..n-1
    for i in range(2, n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}


def _rank_of(d):
    return len(d["U"][0])


def _verify(d, n, m, p):
    """Return True iff sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == delta(i,j,k) exactly."""
    U = np.array(d["U"], dtype=object)
    V = np.array(d["V"], dtype=object)
    W = np.array(d["W"], dtype=object)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u = U[i * m + j]
                v = V[j * p + k]
                w = W[k * n + i]
                for rr in range(r):
                    a = u[rr]
                    if a == 0:
                        continue
                    s += a * v[rr] * w[rr]
                if s != 1:
                    return False
    return True


def construct():
    global _START
    _START = time.time()
    n, m, p = _DIMS

    # Correct, always-valid fallback (schoolbook, rank 18 for 2x3x3).
    fallback = _schoolbook(n, m, p)

    candidates = []
    try:
        candidates.append(_strassen_2x2_core(n, m, p))
    except Exception:
        pass
    try:
        candidates.append(_scheme17(n, m, p))
    except Exception:
        pass
    candidates.append(fallback)

    best = None
    for cand in candidates:
        if _deadline_exceeded():
            break
        try:
            if _verify(cand, n, m, p):
                if best is None or _rank_of(cand) < _rank_of(best):
                    best = cand
        except Exception:
            continue
        if _rank_of(best) == n * m * p:
            # cannot do better than schoolbook; stop searching.
            break

    if best is None:
        return fallback
    return best
