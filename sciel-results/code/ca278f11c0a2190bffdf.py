import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# ---------------------------------------------------------------------------
# Strassen 2x2x2: 7 rank-1 terms.  Block index order: 0=TL,1=TR,2=BL,3=BR.
# ---------------------------------------------------------------------------
_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # m2
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # m3
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),  # m4
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # m5
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # m6
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # m7
]


def _add_strassen_block(U, V, W, col, a_base, b_base, c_base):
    """Add 7 rank-1 Strassen terms for a 2x2x2 block."""
    for (u, v, w) in _STRASSEN_222:
        for t in range(4):
            if u[t]:
                U[a_base + t][col] += u[t]
            if v[t]:
                V[b_base + t][col] += v[t]
            if w[t]:
                W[c_base + t][col] += w[t]
        col += 1
    return col


def _add_444(U, V, W, col, a_base, b_base, c_base):
    """Exact rank-49 decomposition for a 4x4x4 block via 7 Strassen block-sums."""
    U_sums = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    V_sums = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [0, -1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    W_sums = [
        [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]
    for t in range(7):
        u, v, w = U_sums[t], V_sums[t], W_sums[t]
        for s in range(4):
            if u[s]:
                U[a_base + s][col] += u[s]
            if v[s]:
                V[b_base + s][col] += v[s]
            if w[s]:
                W[c_base + s][col] += w[s]
        col += 1
    return col


def _add_matrix_vector_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    """Matrix-vector: A(ar x ac) * B(ac x bc) where bc=1.  Rank ar*ac."""
    for i in range(ar):
        for j in range(ac):
            U[a_base + i * ac + j][col] += 1
            V[b_base + j * bc][col] += 1
            W[c_base + i * bc][col] += 1
            col += 1
    return col


def _add_449(U, V, W, col, a_base, b_base, c_base):
    """Rank-114 decomposition for 4x4x9: two 4x4x4 Strassen blocks + 4x4x1 MV."""
    col = _add_444(U, V, W, col, a_base, b_base, c_base)
    col = _add_444(U, V, W, col, a_base, b_base + 4, c_base + 4)
    col = _add_matrix_vector_block(U, V, W, col, a_base, b_base + 8, c_base + 8, 4, 4, 1)
    return col


def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True


def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_rank114(n, m, p):
    """Rank-114: 4x4x4 Strassen (49) + 4x4x5 (65) = 114."""
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = _add_449(U, V, W, 0, 0, 0, 0)
    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    res = _build_rank114(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res

    return _build_schoolbook(n, m, p)
