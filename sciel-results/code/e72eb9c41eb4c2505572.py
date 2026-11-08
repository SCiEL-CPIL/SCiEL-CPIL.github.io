"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses an optimized block decomposition targeting rank 68.
"""

DIMENSIONS = [3, 3, 8]

import numpy as np


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8

    U_terms = []
    V_terms = []
    W_terms = []

    def add_strassen(row_a_start, col_a_start, row_b_start, col_b_start):
        a = (row_a_start, col_a_start)
        b = (row_a_start, col_a_start + 1)
        c = (row_a_start + 1, col_a_start)
        d = (row_a_start + 1, col_a_start + 1)

        e = (row_b_start, col_b_start)
        f = (row_b_start, col_b_start + 1)
        g = (row_b_start + 1, col_b_start)
        h = (row_b_start + 1, col_b_start + 1)

        c00 = (row_a_start, col_b_start)
        c01 = (row_a_start, col_b_start + 1)
        c10 = (row_a_start + 1, col_b_start)
        c11 = (row_a_start + 1, col_b_start + 1)

        U_terms.append([(a, 1), (d, 1)])
        V_terms.append([(e, 1), (h, 1)])
        W_terms.append([(c00, 1), (c11, 1)])

        U_terms.append([(c, 1), (d, 1)])
        V_terms.append([(e, 1)])
        W_terms.append([(c10, 1), (c11, -1)])

        U_terms.append([(a, 1)])
        V_terms.append([(f, 1), (h, -1)])
        W_terms.append([(c01, 1), (c11, 1)])

        U_terms.append([(d, 1)])
        V_terms.append([(g, 1), (e, -1)])
        W_terms.append([(c00, 1), (c10, 1)])

        U_terms.append([(a, 1), (b, 1)])
        V_terms.append([(h, 1)])
        W_terms.append([(c00, -1), (c01, 1)])

        U_terms.append([(c, 1), (a, -1)])
        V_terms.append([(e, 1), (f, 1)])
        W_terms.append([(c11, 1)])

        U_terms.append([(b, 1), (d, -1)])
        V_terms.append([(g, 1), (h, 1)])
        W_terms.append([(c00, 1)])

    def add_direct(i, j, k):
        U_terms.append([((i, j), 1)])
        V_terms.append([((j, k), 1)])
        W_terms.append([((i, k), 1)])

    # Strassen for A[0:2, 0:2] * B[0:2, col:col+2] for col = 0, 2, 4, 6
    for col in [0, 2, 4, 6]:
        add_strassen(0, 0, 0, col)

    # Direct for A[2, 0:2] * B[0:2, 0:8]
    for j in [0, 1]:
        for k in range(8):
            add_direct(2, j, k)

    # Direct for A[0:2, 2] * B[2, 0:8]
    for i in [0, 1]:
        for k in range(8):
            add_direct(i, 2, k)

    # Direct for A[2, 2] * B[2, 0:8]
    for k in range(8):
        add_direct(2, 2, k)

    rank = len(W_terms)

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r in range(rank):
        for (i, j), val in U_terms[r]:
            U[i * m + j, r] = val
        for (j, k), val in V_terms[r]:
            V[j * p + k, r] = val
        for (i, k), val in W_terms[r]:
            W[k * n + i, r] = val

    def clean(val):
        if abs(val - round(val)) < 1e-9:
            return int(round(val))
        return val

    U_list = [[clean(x) for x in row] for row in U.tolist()]
    V_list = [[clean(x) for x in row] for row in V.tolist()]
    W_list = [[clean(x) for x in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
