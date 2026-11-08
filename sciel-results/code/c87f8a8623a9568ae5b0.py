import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        # M1 = (A11+A22)*(B11+B22)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i_off * m + j_off] = a_scale
        u[(i_off + 1) * m + (j_off + 1)] = a_scale
        v[j_off * p + k_off] = b_scale
        v[(j_off + 1) * p + (k_off + 1)] = b_scale
        w[k_off * n + i_off] = c_scale
        w[(k_off + 1) * n + (i_off + 1)] = c_scale
        add_term(u, v, w)

        # M2 = (A21+A22)*B11
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[(i_off + 1) * m + j_off] = a_scale
        u[(i_off + 1) * m + (j_off + 1)] = a_scale
        v[j_off * p + k_off] = b_scale
        w[k_off * n + (i_off + 1)] = c_scale
        w[(k_off + 1) * n + (i_off + 1)] = -c_scale
        add_term(u, v, w)

        # M3 = A11*(B12-B22)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i_off * m + j_off] = a_scale
        v[j_off * p + (k_off + 1)] = b_scale
        v[(j_off + 1) * p + (k_off + 1)] = -b_scale
        w[(k_off + 1) * n + i_off] = c_scale
        w[(k_off + 1) * n + (i_off + 1)] = c_scale
        add_term(u, v, w)

        # M4 = A22*(B21-B11)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[(i_off + 1) * m + (j_off + 1)] = a_scale
        v[(j_off + 1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off + 1)] = c_scale
        add_term(u, v, w)

        # M5 = (A11+A12)*B22
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off + 1)] = a_scale
        v[(j_off + 1) * p + (k_off + 1)] = b_scale
        w[(k_off + 1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[(i_off + 1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off + 1)] = b_scale
        w[(k_off + 1) * n + (i_off + 1)] = c_scale
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i_off * m + (j_off + 1)] = a_scale
        u[(i_off + 1) * m + (j_off + 1)] = -a_scale
        v[(j_off + 1) * p + k_off] = b_scale
        v[(j_off + 1) * p + (k_off + 1)] = b_scale
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)

    # 4x4 Strassen head: decompose 4x4 x 4x4 as 7x7 = 49 rank-1 terms
    def strassen_4x4_head():
        outer_patterns = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        inner_patterns = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def block_coord(b, l, mat):
            br = b // 2
            bc = b % 2
            lr = l // 2
            lc = l % 2
            r = br * 2 + lr
            c = bc * 2 + lc
            if mat == 'A':
                return r * m + c
            elif mat == 'B':
                return r * p + c
            else:
                return c * n + r

        for op in outer_patterns:
            for ip in inner_patterns:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for (lb, lc) in op[0]:
                    for (ll, lc2) in ip[0]:
                        u[block_coord(lb, ll, 'A')] += lc * lc2
                for (lb, lc) in op[1]:
                    for (ll, lc2) in ip[1]:
                        v[block_coord(lb, ll, 'B')] += lc * lc2
                for (lb, lc) in op[2]:
                    for (ll, lc2) in ip[2]:
                        w[block_coord(lb, ll, 'C')] += lc * lc2
                add_term(u, v, w)

    # Tail: columns 4 and 5 of B handled via Strassen 2x2 blocks
    def strassen_tail_2cols():
        for ib in range(2):
            for jb in range(2):
                strassen_2x2_block(ib * 2, jb * 2, 4)

    # Last column (col 6) handled via schoolbook: 16 terms
    def schoolbook_tail():
        for i in range(4):
            for j in range(4):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head()
    strassen_tail_2cols()
    schoolbook_tail()

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
