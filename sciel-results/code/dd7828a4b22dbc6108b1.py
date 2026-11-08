"""Strassen-block decomposition for 2x5 x 5x5 matrix multiplication (rank < 50)."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _mul_mat(A, B):
    if A is None:
        return B
    if B is None:
        return A
    return A * B


def _build_48():
    """Strassen-style 48-term decomposition (5 = 2+2+1)."""
    # U: rows indexed by (i,j) with i in 0..1, j in 0..4
    U = np.zeros((10, 48), dtype=int)
    # V: rows indexed by (j,k) with j in 0..4, k in 0..4
    V = np.zeros((25, 48), dtype=int)
    # W: rows indexed by (k,i) with k in 0..4, i in 0..1  (W[k*n+i, r])
    W = np.zeros((10, 48), dtype=int)

    def set_u(i, j, t, v):
        U[i * 5 + j, t] += v

    def set_v(j, k, t, v):
        V[j * 5 + k, t] += v

    def set_w(k, i, t, v):
        W[k * 2 + i, t] += v

    # 2x2 block (i in {0,1}, j in {0,1}, k in {0,1}) -> Strassen, terms 0..6
    S = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(0, 1, -1), (1, 1, 1)], [(0, 1, 1), (1, 1, 1)]),
        ([(0, 1, 1), (1, 1, 1)], [(0, 0, 1), (1, 0, 1)], [(0, 0, 1), (1, 0, 1)]),
        ([(1, 0, 1), (1, 1, -1)], [(0, 0, 1), (1, 0, 1)], [(0, 0, 1), (0, 1, 1)]),
        ([(0, 0, 1), (1, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(0, 0, 1), (1, 0, 1)], [(1, 0, 1), (1, 1, 1)]),
        ([(0, 0, -1), (1, 0, 1)], [(0, 1, 1), (1, 1, 1)], [(0, 1, 1), (1, 1, 1)]),
    ]
    for t, (us, vs, ws) in enumerate(S):
        for (i, j, v) in us:
            set_u(i, j, t, v)
        for (j, k, v) in vs:
            set_v(j, k, t, v)
        for (k, i, v) in ws:
            set_w(k, i, t, v)

    # 2x1 blocks (j in {2,3}, k in {0,1}) -> terms 7..14
    for jj in range(2):
        j = 2 + jj
        base = 7 + 4 * jj
        defs = [
            ([(0, j, 1)], [(j, 0, 1)], [(0, 0, 1)]),
            ([(0, j, 1)], [(j, 1, 1)], [(1, 0, 1)]),
            ([(1, j, 1)], [(j, 0, 1)], [(0, 1, 1)]),
            ([(1, j, 1)], [(j, 1, 1)], [(1, 1, 1)]),
        ]
        for t, (us, vs, ws) in enumerate(defs):
            for (i, j2, v) in us:
                set_u(i, j2, base + t, v)
            for (j2, k, v) in vs:
                set_v(j2, k, base + t, v)
            for (k, i, v) in ws:
                set_w(k, i, base + t, v)

    # 1x2 blocks (j in {0,1}, k in {2,3}) -> terms 15..22
    for kk in range(2):
        k = 2 + kk
        base = 15 + 4 * kk
        defs = [
            ([(0, 0, 1)], [(0, k, 1)], [(k, 0, 1)]),
            ([(0, 1, 1)], [(1, k, 1)], [(k, 0, 1)]),
            ([(1, 0, 1)], [(0, k, 1)], [(k, 1, 1)]),
            ([(1, 1, 1)], [(1, k, 1)], [(k, 1, 1)]),
        ]
        for t, (us, vs, ws) in enumerate(defs):
            for (i, j, v) in us:
                set_u(i, j, base + t, v)
            for (j, k2, v) in vs:
                set_v(j, k2, base + t, v)
            for (k2, i, v) in ws:
                set_w(k2, i, base + t, v)

    # 1x1 block (j=4, k=4) -> term 23
    set_u(0, 4, 23, 1)
    set_v(4, 4, 23, 1)
    set_w(4, 0, 23, 1)
    set_u(1, 4, 23, 1)
    set_w(4, 1, 23, 1)

    # 2x1 blocks (j in {2,3}, k=4) -> terms 24..31
    for jj in range(2):
        j = 2 + jj
        base = 24 + 4 * jj
        defs = [
            ([(0, j, 1)], [(j, 4, 1)], [(4, 0, 1)]),
            ([(1, j, 1)], [(j, 4, 1)], [(4, 1, 1)]),
            ([(0, j, 1)], [(j, 4, 1)], [(4, 0, 1)]),
            ([(1, j, 1)], [(j, 4, 1)], [(4, 1, 1)]),
        ]
        for t, (us, vs, ws) in enumerate(defs):
            for (i, j2, v) in us:
                set_u(i, j2, base + t, v)
            for (j2, k, v) in vs:
                set_v(j2, k, base + t, v)
            for (k, i, v) in ws:
                set_w(k, i, base + t, v)

    # 1x2 blocks (j=4, k in {2,3}) -> terms 32..39
    for kk in range(2):
        k = 2 + kk
        base = 32 + 4 * kk
        defs = [
            ([(0, 4, 1)], [(4, k, 1)], [(k, 0, 1)]),
            ([(1, 4, 1)], [(4, k, 1)], [(k, 1, 1)]),
            ([(0, 4, 1)], [(4, k, 1)], [(k, 0, 1)]),
            ([(1, 4, 1)], [(4, k, 1)], [(k, 1, 1)]),
        ]
        for t, (us, vs, ws) in enumerate(defs):
            for (i, j, v) in us:
                set_u(i, j, base + t, v)
            for (j, k2, v) in vs:
                set_v(j, k2, base + t, v)
            for (k2, i, v) in ws:
                set_w(k2, i, base + t, v)

    # 1x1 block (j=4, k=4) -> term 40
    set_u(0, 4, 40, 1)
    set_v(4, 4, 40, 1)
    set_w(4, 0, 40, 1)
    set_u(1, 4, 40, 1)
    set_w(4, 1, 40, 1)

    # 2x1 blocks (j in {0,1}, k=4) -> terms 41..48 (8 terms)
    for jj in range(2):
        j = jj
        base = 41 + 4 * jj
        defs = [
            ([(0, j, 1)], [(j, 4, 1)], [(4, 0, 1)]),
            ([(1, j, 1)], [(j, 4, 1)], [(4, 1, 1)]),
            ([(0, j, 1)], [(j, 4, 1)], [(4, 0, 1)]),
            ([(1, j, 1)], [(j, 4, 1)], [(4, 1, 1)]),
        ]
        for t, (us, vs, ws) in enumerate(defs):
            for (i, j2, v) in us:
                set_u(i, j2, base + t, v)
            for (j2, k, v) in vs:
                set_v(j2, k, base + t, v)
            for (k, i, v) in ws:
                set_w(k, i, base + t, v)

    return U, V, W


def _is_valid(U, V, W):
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    # bilinear identity: (C)_{i,j} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for t in range(U.shape[1]):
                    a = U[i * m + j, t]
                    b = V[j * p + k, t]
                    c = W[k * n + i, t]
                    if a == 0 or b == 0 or c == 0:
                        continue
                    acc += a * b * c
                expected = 1 if (i == 0 and j == k) else 0
                if acc != expected:
                    return False
    return True


def _to_list(M):
    return [[int(x) for x in row] for row in np.asarray(M, dtype=object)]


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _maximize_rank(U, V, W, deadline):
    """Keep a maximal subset of rank-1 terms that still satisfies the identity."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)

    # contribution of term t to entry (i,j,k)
    contrib = np.zeros((n, m, p, R), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for t in range(R):
                    contrib[i, j, k, t] = U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]

    active = list(range(R))
    # greedy drop: repeatedly remove a term whose removal keeps the identity
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for t in list(active):
            if time.time() >= deadline:
                break
            active2 = [x for x in active if x != t]
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        acc = 0
                        for x in active2:
                            c = contrib[i, j, k, x]
                            if c != 0:
                                acc += c
                        expected = 1 if (i == 0 and j == k) else 0
                        if acc != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                active = active2
                changed = True
    keep = set(active)
    U2 = np.zeros((n * m, len(active)), dtype=object)
    V2 = np.zeros((m * p, len(active)), dtype=object)
    W2 = np.zeros((p * n, len(active)), dtype=object)
    for new_t, old_t in enumerate(active):
        U2[:, new_t] = U[:, old_t]
        V2[:, new_t] = V[:, old_t]
        W2[:, new_t] = W[:, old_t]
    return U2, V2, W2


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin under the 150s kill

    U, V, W = _build_48()
    if _is_valid(U, V, W) and time.time() < deadline:
        U, V, W = _maximize_rank(U, V, W, deadline)
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # fallback: guaranteed schoolbook (rank 50)
    return _schoolbook()
# EVOLVE-BLOCK-END
