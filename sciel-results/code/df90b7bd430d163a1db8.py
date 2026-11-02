"""
Rank-49 exact tensor decomposition for 4x4 x 4x4 matrix multiplication.

Recursive Strassen construction (rank 49) with verification and fallback.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# 2x2 Strassen patterns: 7 products
# Each entry: (u_entries, v_entries, w_entries)
# entries are lists of (row, col, coeff)
_S2X2 = [
    # M1 = (a+d)(e+h)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # M2 = (c+d)e
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
    # M3 = a(f-h)
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
    # M4 = d(g-e)
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
    # M5 = (a+b)h
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(0,0,-1)]),
    # M6 = (c-a)(e+f)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    # M7 = (b-d)(g+h)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]


def _build_strassen_4x4(n, m, p):
    """
    Construct rank-49 decomposition via recursive Strassen.
    Returns U (n*m x 49), V (m*p x 49), W (p*n x 49).
    W uses trace convention: W[k*n + i, r].
    """
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    for s, (outer_u, outer_v, outer_w) in enumerate(_S2X2):
        for t, (inner_u, inner_v, inner_w) in enumerate(_S2X2):
            r = s * 7 + t

            # U: A[row, col] -> index row*m + col
            for (P, Q, sa) in outer_u:
                for (ir, jc, sx) in inner_u:
                    c = sa * sx
                    if c != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        U[row * m + col, r] += c

            # V: B[row, col] -> index row*p + col
            for (P, Q, sb) in outer_v:
                for (ir, jc, sy) in inner_v:
                    c = sb * sy
                    if c != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        V[row * p + col, r] += c

            # W: C[i, j] -> trace convention W[j*n + i, r]
            for (P, Q, sc) in outer_w:
                for (ir, jc, sz) in inner_w:
                    c = sc * sz
                    if c != 0:
                        i = 2 * P + ir
                        j = 2 * Q + jc
                        W[j * n + i, r] += c

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank).astype(np.float64)
    V3 = V.reshape(m, p, rank).astype(np.float64)
    W3 = W.reshape(p, n, rank).astype(np.float64)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=1e-9))


def _schoolbook(n, m, p):
    """Rank n*m*p trivial decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 115.0
    n, m, p = DIMENSIONS

    # Primary: rank-49 recursive Strassen
    try:
        U, V, W = _build_strassen_4x4(n, m, p)
        if time.time() < deadline:
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    if time.time() < deadline:
        _verify(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
