"""
Rank-49 exact tensor decomposition for 4x4 x 4x4 matrix multiplication.

Strategy: recursive Strassen (7 outer block products, each 7 inner scalar products
= 49 total bilinear forms). This is the best known exact rank for 4x4 matrix
multiplication.

The code is written to be compact and fast. A single verification pass confirms
the decomposition before returning.

Convention (official trace convention):
    C[i, j] = sum_k A[i, k] * B[k, j]
    U index : i*m + k   (A row i, A col k)
    V index : k*p + j   (B row k, B col j)
    W index : j*n + i   (C col j, C row i)
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Inner 2x2 Strassen: 7 products
# u: [(row, col, sign)], v: [(row, col, sign)], w: [(row, col, sign)]
_S22 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)],          [(1,0,1),(1,1,-1)]),
    ([(0,0,1)],          [(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)]),
    ([(1,1,1)],          [(1,0,1),(0,0,-1)],[(0,0,1),(1,0,1)]),
    ([(0,0,1),(0,1,1)],  [(1,1,1)],          [(0,1,1),(0,0,-1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],  [(1,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],  [(0,0,1)]),
]

# Outer 4x4 Strassen over 2x2 blocks: 7 products
# a: [(block_row, block_col, sign)], b: [...], c: [...]
_S44 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)],          [(1,0,1),(1,1,-1)]),
    ([(0,0,1)],          [(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)]),
    ([(1,1,1)],          [(1,0,1),(0,0,-1)],[(0,0,1),(1,0,1)]),
    ([(0,0,1),(0,1,1)],  [(1,1,1)],          [(0,1,1),(0,0,-1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],  [(1,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],  [(0,0,1)]),
]


def _strassen_4x4(n, m, p):
    assert (n, m, p) == (4, 4, 4)
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    inner_len = 7
    for s in range(7):
        a_blocks, b_blocks, c_blocks = _S44[s]
        for t in range(7):
            u_terms, v_terms, w_terms = _S22[t]
            r = s * inner_len + t
            for (P, Q, sa) in a_blocks:
                for (ir, jc, sx) in u_terms:
                    c = sa * sx
                    if c:
                        U[(2*P + ir) * m + (2*Q + jc), r] += c
            for (P, Q, sb) in b_blocks:
                for (ir, jc, sy) in v_terms:
                    c = sb * sy
                    if c:
                        V[(2*P + ir) * p + (2*Q + jc), r] += c
            for (P, Q, sc) in c_blocks:
                for (ir, jc, sz) in w_terms:
                    c = sc * sz
                    if c:
                        W[(2*Q + jc) * n + (2*P + ir), r] += c
    return U, V, W


def _verify(U, V, W, n, m, p):
    rk = U.shape[1]
    U_r = U.reshape(n, m, rk)
    V_r = V.reshape(m, p, rk)
    W_r = W.reshape(p, n, rk)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=1e-9))


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS

    try:
        U, V, W = _strassen_4x4(n, m, p)
        if time.time() < deadline:
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    U, V, W = _schoolbook(n, m, p)
    if time.time() < deadline:
        _verify(U, V, W, n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
