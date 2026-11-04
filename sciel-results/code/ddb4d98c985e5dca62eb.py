"""
Matrix multiplication 4x4x6 tensor decomposition.
Uses a mixed approach: 4x4x4 Strassen (rank 49) for the first 4 columns
and four independent 2x2x2 Strassen blocks (rank 7 each = 28 total)
for the last 2 columns of B. Total rank = 77.
"""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np


def _strassen_2x2():
    """Return 2x2x2 Strassen factors as lists. U:(4,7), V:(4,7), W:(4,7)."""
    r = 7
    U = [[0.0] * r for _ in range(4)]
    V = [[0.0] * r for _ in range(4)]
    W = [[0.0] * r for _ in range(4)]
    prods = [
        ([(0, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0), (1, 1, 1.0)]),
        ([(1, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0)], [(1, 0, 1.0), (1, 1, -1.0)]),
        ([(0, 0, 1.0)], [(0, 1, 1.0), (1, 1, -1.0)], [(0, 1, 1.0), (1, 1, 1.0)]),
        ([(1, 1, 1.0)], [(1, 0, 1.0), (0, 0, -1.0)], [(0, 0, 1.0), (1, 0, 1.0)]),
        ([(0, 0, 1.0), (0, 1, 1.0)], [(1, 1, 1.0)], [(0, 0, -1.0), (0, 1, 1.0)]),
        ([(1, 0, 1.0), (0, 0, -1.0)], [(0, 0, 1.0), (0, 1, 1.0)], [(1, 1, 1.0)]),
        ([(0, 1, 1.0), (1, 1, -1.0)], [(1, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0)]),
    ]
    for c, (ub, vb, wb) in enumerate(prods):
        for (i, j, cv) in ub:
            U[i * 2 + j][c] += cv
        for (j, k, cv) in vb:
            V[j * 2 + k][c] += cv
        for (i, k, cv) in wb:
            W[k * 2 + i][c] += cv
    return U, V, W


def _kron444(Us, Vs, Ws):
    """Build 4x4x4 Strassen factors via Kronecker product of 2x2 factors."""
    r = 49
    U4 = [[0.0] * r for _ in range(16)]
    V4 = [[0.0] * r for _ in range(16)]
    W4 = [[0.0] * r for _ in range(16)]

    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7 * r1 + r2
                            U4[i * 4 + j][col] = Us[i1 * 2 + j1][r1] * Us[i2 * 2 + j2][r2]

    for j1 in range(2):
        for j2 in range(2):
            j = 2 * j1 + j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7 * r1 + r2
                            V4[j * 4 + k][col] = Vs[j1 * 2 + k1][r1] * Vs[j2 * 2 + k2][r2]

    for k1 in range(2):
        for k2 in range(2):
            k = 2 * k1 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7 * r1 + r2
                            W4[k * 4 + i][col] = Ws[k1 * 2 + i1][r1] * Ws[k2 * 2 + i2][r2]

    return U4, V4, W4


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 145.0

    if time.time() > deadline - 5.0:
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

    Us, Vs, Ws = _strassen_2x2()
    U4, V4, W4 = _kron444(Us, Vs, Ws)

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    # Part 1: 4x4x4 Strassen
    for idx in range(16):
        for c in range(rank_444):
            if U4[idx][c] != 0:
                U[idx][c] = U4[idx][c]

    for j in range(4):
        for k in range(4):
            for c in range(rank_444):
                if V4[j * 4 + k][c] != 0:
                    V[j * 6 + k][c] = V4[j * 4 + k][c]

    for k in range(4):
        for i in range(4):
            for c in range(rank_444):
                if W4[k * 4 + i][c] != 0:
                    W[k * 4 + i][c] = W4[k * 4 + i][c]

    # Part 2: Four 2x2x2 Strassen blocks for columns 4-5
    blocks = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for b_idx, (I, J) in enumerate(blocks):
        col_off = rank_444 + b_idx * 7
        j_base = 2 * J
        i_base = 2 * I
        k_base = 4

        for i0 in range(2):
            for j0 in range(2):
                i_glob = i_base + i0
                j_glob = j_base + j0
                for s in range(7):
                    v = Us[i0 * 2 + j0][s]
                    if v != 0:
                        U[i_glob * 4 + j_glob][col_off + s] += v

        for j0 in range(2):
            for k0 in range(2):
                j_glob = j_base + j0
                k_glob = k_base + k0
                for s in range(7):
                    v = Vs[j0 * 2 + k0][s]
                    if v != 0:
                        V[j_glob * 6 + k_glob][col_off + s] += v

        for k0 in range(2):
            for i0 in range(2):
                i_glob = i_base + i0
                k_glob = k_base + k0
                for s in range(7):
                    v = Ws[k0 * 2 + i0][s]
                    if v != 0:
                        W[k_glob * 4 + i_glob][col_off + s] += v

    if time.time() > deadline - 8.0:
        return {"U": U, "V": V, "W": W}

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, total_rank)
    Vr = Va.reshape(m, p, total_rank)
    Wr = Wa.reshape(p, n, total_rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = float(np.max(np.abs(T - 1.0)))

    if err > 1e-6:
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

    return {"U": U, "V": V, "W": W}
