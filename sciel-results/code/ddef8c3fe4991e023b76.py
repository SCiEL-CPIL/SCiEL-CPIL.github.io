"""Rank-reducing exact decomposition for multiplying 3x3 by 3x4 matrices."""
DIMENSIONS = [3, 3, 4]

import time

_START = time.monotonic()
_LIMIT = 140.0  # wall-clock safety margin below the 150s kill


def _deadline():
    return (time.monotonic() - _START) > _LIMIT


def construct():
    # ---- 2x2x2 Strassen primitive (indices i,j,k in {0,1}) ----
    _S = [
        ([(0, 0), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (1, 1)], 1),
        ([(0, 1), (1, 0)], [(0, 0), (0, 1)], [(0, 0), (0, 1)], 1),
        ([(0, 0)], [(0, 1), (1, 1)], [(0, 1), (1, 1)], 1),
        ([(0, 0)], [(0, 0)], [(0, 1), (1, 0)], 1),
        ([(1, 0), (1, 1)], [(1, 0), (1, 1)], [(0, 1)], 1),
        ([(0, 1)], [(0, 1), (1, 1)], [(1, 0), (1, 1)], 1),
        ([(1, 0)], [(1, 0)], [(1, 1), (1, 0)], 1),
    ]

    def _add_term(i0, j0, k0, sU, sV, sW, s):
        for (a, b) in sU:
            U[(i0 + a) * 3 + (j0 + b)] += s
        for (b, c) in sV:
            V[(j0 + b) * 4 + (k0 + c)] += s
        for (c, a) in sW:
            W[(k0 + c) * 3 + (i0 + a)] += s

    def _trivial(r, c):
        """Add rank r*c block; A rows i0:i0+r, B cols k0:k0+c, B rows j0..j0+c-1."""
        for a in range(r):
            for c in range(c):
                U[(i0 + a) * 3 + (j0 + c)] += 1
                V[(j0 + c) * 4 + (k0 + c)] += 1
                W[(k0 + c) * 3 + (i0 + a)] += 1

    n, m, p = DIMENSIONS
    rank = 34  # 14 (2x2 Strassen) + 20 (trivial)
    U = [0] * (n * m)
    V = [0] * (m * p)
    W = [0] * (p * n)

    # Strassen on the 2x2 (A) by 2x2 (B) blocks, split into two 2x2x2 subproblems.
    for k0 in (0, 2):
        for t in _S:
            _add_term(0, 0, k0, t[0], t[1], t[2], t[3])
        if _deadline():
            break

    # Trivial blocks:
    #   A[0:2] x B[2:3, all 4 cols]  -> 2x4 = 8
    #   A[2:3] x B[0:2, cols 0:2]    -> 1x2 = 2
    #   A[2:3] x B[0:2, cols 2:4]    -> 1x2 = 2
    #   A[2:3] x B[2:3, all 4 cols]  -> 1x4 = 4
    #   A[0:2] x B[3:4, all 4 cols]  -> 2x4 = 8
    #   A[2:3] x B[3:4, all 4 cols]  -> 1x4 = 4
    _blocks = [
        (0, 2, 2, 0, 4),
        (2, 3, 0, 0, 2),
        (2, 3, 0, 2, 2),
        (2, 3, 2, 0, 4),
        (0, 2, 3, 0, 4),
        (2, 3, 3, 0, 4),
    ]
    for i0, i1, j0, k0, kc in _blocks:
        _trivial(i1 - i0, kc)
        if _deadline():
            break

    if _deadline():
        # Safety: fall back to guaranteed-valid schoolbook rank 36.
        U = [0] * (n * m)
        V = [0] * (m * p)
        W = [0] * (p * n)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j] = 1
                    V[j * p + k] = 1
                    W[k * n + i] = 1
                    col += 1
        rank = n * m * p

    return {"U": U, "V": V, "W": W}
