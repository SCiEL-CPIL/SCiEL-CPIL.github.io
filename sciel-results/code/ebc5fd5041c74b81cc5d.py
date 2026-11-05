import numpy as np
import time

_START = time.time()
_DEADLINE = 120.0


def construct():
    n, m, p = 4, 5, 5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(np.array(u_vec, dtype=float))
        V_cols.append(np.array(v_vec, dtype=float))
        W_cols.append(np.array(w_vec, dtype=float))

    def add_term(i, k, j, cu=1.0, cv=1.0, cw=1.0):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        u[i * m + k] = cu
        v[k * p + j] = cv
        w[j * n + i] = cw
        add_rank1(u, v, w)

    # Part 1: 4x4 * 4x4 subproblem (Strassen rank 49)
    # Outer Strassen 7 products, each with inner Strassen 7 products = 49 terms

    # Outer Strassen: combine 2x2 blocks of 2x2 matrices
    outer_A = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    outer_B = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    outer_C = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    def a_glob(block, li, lk):
        return (li + 2 * (block >> 1), lk + 2 * (block & 1))

    def b_glob(block, lk, lj):
        return (lk + 2 * (block >> 1), lj + 2 * (block & 1))

    def c_glob(block, li, lj):
        return (li + 2 * (block >> 1), lj + 2 * (block & 1))

    # Inner Strassen for 2x2 * 2x2
    inner_A = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    inner_B = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    inner_C = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    for o in range(7):
        for i in range(7):
            u = np.zeros(n * m, dtype=float)
            v = np.zeros(m * p, dtype=float)
            w = np.zeros(p * n, dtype=float)

            for (blk, sgn) in outer_A[o].items():
                for (li, lk, isgn) in inner_A[i]:
                    gi, gk = a_glob(blk, li, lk)
                    u[gi * m + gk] += sgn * isgn

            for (blk, sgn) in outer_B[o].items():
                for (lk, lj, isgn) in inner_B[i]:
                    gk, gj = b_glob(blk, lk, lj)
                    v[gk * p + gj] += sgn * isgn

            for (blk, sgn) in outer_C[o].items():
                for (li, lj, isgn) in inner_C[i]:
                    gi, gj = c_glob(blk, li, lj)
                    w[gj * n + gi] += sgn * isgn

            add_rank1(u, v, w)

    # Part 2: 4x4 * 4x1 (schoolbook, 16 terms)
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4)

    # Part 3: 4x1 * 1x5 (schoolbook, 20 terms)
    for i in range(4):
        for j in range(5):
            add_term(i, 4, j)

    rank = len(U_cols)

    if rank == 0:
        add_term(0, 0, 0)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    elapsed = time.time() - _START
    if elapsed > _DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
