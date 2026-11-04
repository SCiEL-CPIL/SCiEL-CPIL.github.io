"""Matrix multiplication 4x5x6 rank minimization.

Block decomposition:
  rows 4 = 2+2, middle 5 = 2+3, cols 6 = 3+3
  -> 4 blocks M_{2,2,3}  (rank 11 each)  + 4 blocks M_{2,3,3} (rank 17 each)
  Total rank = 4*11 + 4*17 = 112  (schoolbook is 120).

M_{2,2,3} = M_{2,2,1} (rank 4) + M_{2,2,2} (Strassen rank 7) = 11
M_{2,3,3} = M_{2,1,3} (rank 6) + M_{2,2,3} (rank 11)         = 17
"""
import time

DIMENSIONS = [4, 5, 6]


def _strassen_222():
    """Rank-7 exact decomposition of M_{2,2,2}.

    Returns (U,V,W) each of shape (4,7). Local row index = i*2 + j (U),
    j*2 + k (V), k*2 + i (W); i,j,k in {0,1}.
    Each column is one rank-1 tensor u* v* w.
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 1],
        [1, 0, -1, 0, 0, 0, 0],
        [0, 1, 0, -1, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, -1, 1, 1, 0, 0, 0],
    ]
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition of M_{n,m,p}, rank n*m*p, entries 0/1."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _place(global_U, global_V, global_W, n, m, p, n0, j0, k0, U, V, W, col0):
    """Embed a local M_{n,m,p} decomposition at global offsets (n0,j0,k0),
    starting at global column col0. Returns the next free column index."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            src = U[i * m + j]
            dst = global_U[(i + n0) * m + (j + j0)]
            for c in range(r):
                dst[col0 + c] = src[c]
    for j in range(m):
        for k in range(p):
            src = V[j * p + k]
            dst = global_V[(j + j0) * p + (k + k0)]
            for c in range(r):
                dst[col0 + c] = src[c]
    for k in range(p):
        for i in range(n):
            src = W[k * n + i]
            dst = global_W[(k + k0) * n + (i + n0)]
            for c in range(r):
                dst[col0 + c] = src[c]
    return col0 + r


def _build():
    n, m, p = 4, 5, 6

    # --- Base decompositions (local) ---
    sU, sV, sW = _strassen_222()          # M_{2,2,2}, rank 7
    aU, aV, aW = _schoolbook(2, 2, 1)     # M_{2,2,1}, rank 4
    bU, bV, bW = _schoolbook(2, 1, 3)     # M_{2,1,3}, rank 6

    # --- M_{2,2,3} = M_{2,2,1} (cols 0..2) + M_{2,2,2} (col 2) , rank 11 ---
    c223 = [[0] * 11 for _ in range(4)]   # n*m = 4
    v223 = [[0] * 11 for _ in range(6)]   # m*p = 6
    w223 = [[0] * 11 for _ in range(6)]   # p*n = 6
    _place(c223, v223, w223, 2, 2, 1, 0, 0, 0, aU, aV, aW, 0)
    _place(c223, v223, w223, 2, 2, 2, 0, 0, 2, sU, sV, sW, 4)

    # --- M_{2,3,3} = M_{2,1,3} (mid col 0) + M_{2,2,3} (mid col 1), rank 17 ---
    c233 = [[0] * 17 for _ in range(6)]   # n*m = 6
    v233 = [[0] * 17 for _ in range(9)]   # m*p = 9
    w233 = [[0] * 17 for _ in range(6)]   # p*n = 6
    _place(c233, v233, w233, 2, 1, 3, 0, 0, 0, bU, bV, bW, 0)
    _place(c233, v233, w233, 2, 2, 3, 0, 1, 0, c223, v223, w223, 6)

    # --- Global assembly ---
    total = 4 * 11 + 4 * 17               # = 112
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]
    col = 0
    for i0 in (0, 2):
        for j0 in (0, 2):
            for k0 in (0, 3):
                if j0 == 0:
                    col = _place(U, V, W, 2, 2, 3, i0, j0, k0, c223, v223, w223, col)
                else:
                    col = _place(U, V, W, 2, 3, 3, i0, j0, k0, c233, v233, w233, col)
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    try:
        result = _build()
        if time.time() > deadline:
            raise RuntimeError("deadline")
        return result
    except Exception:
        # Fallback: guaranteed-correct schoolbook decomposition (rank 120).
        n, m, p = DIMENSIONS
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
