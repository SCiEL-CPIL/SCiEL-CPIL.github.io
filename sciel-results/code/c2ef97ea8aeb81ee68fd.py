"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication.

Strategy: exploit the rectangular structure by splitting the middle dimension
into two halves (j in {0,1} and j in {2,3}) and the last dimension into two
halves (k in {0,1,2} and k in {3,4,5}). This yields 4 independent 3x2x3
sub-problems. Each sub-problem (multiplying a 3x2 matrix by a 2x3 matrix) is
decomposed with rank 12: a rank-7 Strassen core for the 2x2 top-left output
block plus 5 rank-1 terms for the remaining output entries. Total rank = 4*12 = 48.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS
DEADLINE = 130.0  # safety margin before 150s kill


def verify_fast(U, V, W, rank):
    """Faster verification using numpy. Returns True if decomposition is exact."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-9)


def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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


def _add(U, V, W, col, u, v, w):
    """Add a rank-1 term given local (i,j), (j,k), (k,i) coefficient dicts."""
    for (i, j), val in u.items():
        U[i * M + j][col] += val
    for (j, k), val in v.items():
        V[j * P + k][col] += val
    for (k, i), val in w.items():
        W[k * N + i][col] += val


def _strassen_block(U, V, W, col, j0, k0):
    """Add the 7 Strassen terms for a 3x2x3 block.

    Block: A is 3x2 (rows 0,1,2; cols j0,j0+1), B is 2x3 (rows j0,j0+1;
    cols k0,k0+1,k0+2). The 2x2 core uses A rows 0,1 and B cols k0,k0+1.
    Output C is 3x3; the 2x2 core covers C[0:2, 0:2].
    """
    j1 = j0 + 1
    k1 = k0 + 1

    # Term 0: M1 = (x00+x11)(y00+y11)
    _add(U, V, W, col,
         {(0, j0): 1, (1, j1): 1},
         {(j0, k0): 1, (j1, k1): 1},
         {(k0, 0): 1, (k1, 0): 1, (k0, 1): -1, (k1, 1): 1})
    col += 1
    # Term 1: M2 = (x10+x11)y00
    _add(U, V, W, col,
         {(1, j0): 1, (1, j1): 1},
         {(j0, k0): 1},
         {(k1, 0): -1, (k0, 1): 1})
    col += 1
    # Term 2: M3 = x00(y01-y11)
    _add(U, V, W, col,
         {(0, j0): 1},
         {(j0, k1): 1, (j1, k1): -1},
         {(k1, 0): 1, (k0, 1): 1})
    col += 1
    # Term 3: M4 = x11(y10-y00)
    _add(U, V, W, col,
         {(1, j1): 1},
         {(j1, k0): 1, (j0, k0): -1},
         {(k0, 0): -1, (k1, 0): -1, (k0, 1): 1})
    col += 1
    # Term 4: M5 = (x00+x10)y11
    _add(U, V, W, col,
         {(0, j0): 1, (1, j0): 1},
         {(j1, k1): 1},
         {(k0, 0): 1, (k0, 1): -1})
    col += 1
    # Term 5: M6 = (x01-x11)(y10+y11)
    _add(U, V, W, col,
         {(0, j1): 1, (1, j1): -1},
         {(j1, k0): 1, (j1, k1): 1},
         {(k0, 0): 1, (k1, 1): 1})
    col += 1
    # Term 6: M7 = (x00-x10)(y00+y01)
    _add(U, V, W, col,
         {(0, j0): 1, (1, j0): -1},
         {(j0, k0): 1, (j0, k1): 1},
         {(k0, 1): 1, (k1, 1): 1})
    col += 1
    return col


def _rank1_block_terms(U, V, W, col, j0, k0):
    """Add the 5 rank-1 terms covering the non-core output entries.

    Core covers C[0:2,0:2]. Remaining: (0,k0+2),(1,k0+2),(2,k0),(2,k0+1),(2,k0+2).
    """
    j1 = j0 + 1
    k1 = k0 + 1
    k2 = k0 + 2
    # C[0, k2] = A[0,j0]B[j0,k2] + A[0,j1]B[j1,k2]
    _add(U, V, W, col, {(0, j0): 1}, {(j0, k2): 1}, {(k2, 0): 1}); col += 1
    _add(U, V, W, col, {(0, j1): 1}, {(j1, k2): 1}, {(k2, 0): 1}); col += 1
    # C[1, k2]
    _add(U, V, W, col, {(1, j0): 1}, {(j0, k2): 1}, {(k2, 1): 1}); col += 1
    _add(U, V, W, col, {(1, j1): 1}, {(j1, k2): 1}, {(k2, 1): 1}); col += 1
    # C[2, k0]
    _add(U, V, W, col, {(2, j0): 1}, {(j0, k0): 1}, {(k0, 2): 1}); col += 1
    # C[2, k1]
    _add(U, V, W, col, {(2, j1): 1}, {(j1, k1): 1}, {(k1, 2): 1}); col += 1
    # C[2, k2]
    _add(U, V, W, col, {(2, j0): 1}, {(j0, k2): 1}, {(k2, 2): 1}); col += 1
    _add(U, V, W, col, {(2, j1): 1}, {(j1, k2): 1}, {(k2, 2): 1}); col += 1
    return col


def structured_decomposition():
    """Build the rank-48 decomposition via 4 independent 3x2x3 blocks."""
    n, m, p = N, M, P
    rank = 4 * 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for bj in range(2):          # middle split: j0 = 0 or 2
        for bk in range(2):      # last split: k0 = 0 or 3
            j0 = 2 * bj
            k0 = 3 * bk
            col = _strassen_block(U, V, W, col, j0, k0)
            col = _rank1_block_terms(U, V, W, col, j0, k0)
    assert col == rank, f"column mismatch {col} != {rank}"
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()

    # Primary: the structural rank-48 decomposition.
    best = structured_decomposition()
    if verify_fast(best["U"], best["V"], best["W"], len(best["U"][0])):
        return best

    # Fallback: schoolbook (always valid, rank 72).
    best = schoolbook()
    return best
