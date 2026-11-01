import numpy as np

def construct():
    n, m, p = 2, 2, 14
    nm = n * m
    mp = m * p
    pn = p * n

    num_blocks = p // 2
    rank = num_blocks * 7

    # Strassen's 7 terms for 2x2 * 2x2
    # Each term: (a_idx, a_val, b_idx, b_val, c_idx, c_val)
    # Local indices: 0=A00, 1=A01, 2=A10, 3=A11 for A
    #               0=B00, 1=B01, 2=B10, 3=B11 for B
    #               c_idx: 0=C00, 1=C01, 2=C10, 3=C11
    # c_idx mapping: c_idx = i * 2 + k where (i,k) is row/col of C
    strassen_terms = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]

    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            a_il, a_vl, b_il, b_vl, c_il, c_vl = strassen_terms[s]

            for ai, av in zip(a_il, a_vl):
                a_i = ai // 2
                a_j = ai % 2
                U[a_i * m + a_j, col] = av

            for bi, bv in zip(b_il, b_vl):
                b_j = bi // 2
                b_k_local = bi % 2
                b_k_global = 2 * t + b_k_local
                V[b_j * p + b_k_global, col] = bv

            for ci, cv in zip(c_il, c_vl):
                c_i = ci // 2
                c_k_local = ci % 2
                c_k_global = 2 * t + c_k_local
                W[c_k_global * n + c_i, col] = cv

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
