"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

_START = time.perf_counter()
_DEADLINE = 120.0
_TIME_MARGIN = 25.0


def _remaining():
    return _DEADLINE - _TIME_MARGIN - (time.perf_counter() - _START)


def _verify(U, V, W, n=2, m=2, p=6):
    rank = len(U[0])
    U3 = np.array(U, dtype=float).reshape(n, m, rank)
    V3 = np.array(V, dtype=float).reshape(m, p, rank)
    W3 = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ikr,kjr,jir->ikj', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=1e-8))


def _as_ints(M, tol=1e-6):
    A = np.array(M, dtype=float)
    R = np.round(A)
    if np.allclose(A, R, atol=tol):
        return [[int(x) for x in row] for row in R]
    return M


def _strassen_block_coeffs():
    # Each entry: (r, a_row, b_col, c_row, coeff). a_row in 0..3 (A[i,k] row i*m+k),
    # b_col in 0..5 (B[k,j] col j), c_row in 0..5 (C row j*n+i).
    return [
        (0, 0, 0, 0, 1), (0, 3, 0, 0, 1), (0, 0, 0, 5, 1), (0, 3, 0, 5, 1),
        (1, 2, 0, 1, 1), (1, 3, 0, 1, 1), (1, 0, 0, 1, -1), (1, 3, 0, 5, -1),
        (2, 0, 0, 2, 1), (2, 0, 1, 2, -1), (2, 0, 0, 5, 1), (2, 0, 1, 5, -1),
        (3, 3, 0, 0, 1), (3, 3, 1, 0, -1), (3, 3, 0, 1, 1), (3, 3, 1, 1, -1),
        (4, 0, 0, 0, 1), (4, 1, 0, 0, 1), (4, 0, 0, 2, -1), (4, 1, 0, 2, -1),
        (5, 2, 0, 5, 1), (5, 0, 0, 5, -1), (5, 2, 1, 5, 1), (5, 0, 1, 5, -1),
        (6, 1, 0, 0, 1), (6, 3, 0, 0, -1), (6, 1, 1, 0, 1), (6, 3, 1, 0, -1),
    ]


def _build_rank21(n=2, m=2, p=6):
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    base = _strassen_block_coeffs()
    for b in range(3):
        for (r, a_row, b_col, c_row, c) in base:
            rr = 7 * b + r
            U[a_row][rr] += c
            V[b_col][rr] += c
            W[c_row][rr] += c
    return U, V, W


def _build_schoolbook(n=2, m=2, p=6):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][idx] = 1
                V[k * p + j][idx] = 1
                W[j * n + i][idx] = 1
                idx += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank21(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": _as_ints(U), "V": _as_ints(V), "W": _as_ints(W)}
    if _remaining() <= 0.0:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
