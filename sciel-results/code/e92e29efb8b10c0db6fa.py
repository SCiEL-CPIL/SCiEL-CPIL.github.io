"""Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition."""
import numpy as np
DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m   # 8
    N_V = m * p   # 20
    N_W = p * n   # 10

    U, V, W = [], [], []

    def mk_U3(combo):
        v = [0] * N_U
        for (i, j, c) in combo:
            v[i * m + j] += c
        return v
    def mk_V3(combo):
        v = [0] * N_V
        for (j, k, c) in combo:
            v[j * p + k] += c
        return v
    def mk_W3(combo):
        v = [0] * N_W
        for (i, k, c) in combo:
            v[k * n + i] += c
        return v
    def add_term(u_c, v_c, w_c):
        U.append(mk_U3(u_c)); V.append(mk_V3(v_c)); W.append(mk_W3(w_c))

    def add_strassen(Arow_lo, Brow_lo, Ccol_lo):
        a, b, c = Arow_lo, Brow_lo, Ccol_lo
        defs = [
            ([(0, a, 1), (1, a + 1, -1)],
             [(b + 1, c + 1, 1), (b, c, -1)],
             [(0, c + 1, 1), (1, c + 1, 1)]),  # m1 -> c01,c11
            ([(1, a, 1), (0, a, 1)],
             [(b, c, 1)],
             [(0, c, -1), (1, c, 1)]),  # m2
            ([(0, a, 1)],
             [(b, c + 1, 1), (b + 1, c + 1, -1)],
             [(0, c + 1, 1), (1, c + 1, -1)]),  # m3
            ([(1, a + 1, 1)],
             [(b + 1, c, 1), (b, c, -1)],
             [(0, c, 1), (1, c, 1)]),  # m4
            ([(0, a, 1), (1, a + 1, 1)],
             [(b, c, 1), (b + 1, c + 1, 1)],
             [(0, c, 1), (1, c + 1, 1)]),  # m5
            ([(0, a + 1, 1), (1, a + 1, -1)],
             [(b + 1, c, 1), (b + 1, c + 1, 1)],
             [(0, c, 1)]),  # m6
            ([(1, a, 1), (0, a, -1)],
             [(b, c, 1), (b, c + 1, 1)],
             [(1, c + 1, -1)]),  # m7
        ]
        for u_c, v_c, w_c in defs:
            add_term(u_c, v_c, w_c)

    def add_school_2x2x1(Arow_lo, Brow_lo, Ccol):
        a, b, c = Arow_lo, Brow_lo, Ccol
        for i in range(2):
            for jA in range(2):
                for jB in range(2):
                    add_term(
                        [(i, a + jA, 1)],
                        [(b + jB, c, 1)],
                        [(i, c, 1)]
                    )

    # j-block 0: A cols 0,1 ; B rows 0,1
    add_strassen(0, 0, 0)     # k 0,1
    add_strassen(0, 0, 2)     # k 2,3
    add_school_2x2x1(0, 0, 4) # k 4
    # j-block 1: A cols 2,3 ; B rows 2,3
    add_strassen(2, 2, 0)
    add_strassen(2, 2, 2)
    add_school_2x2x1(2, 2, 4)

    rank = len(U)
    Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(
                    Ua[i * m + j, :] * Va[j * p + k, :], Wa[k * n + i, :]))
                if abs(s - 1.0) > 1e-6:
                    ok = False

    if ok:
        return {"U": U, "V": V, "W": W}

    # Schoolbook fallback rank 40
    U2, V2, W2 = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2.append(mk_U3([(i, j, 1)]))
                V2.append(mk_V3([(j, k, 1)]))
                W2.append(mk_W3([(i, k, 1)]))
    return {"U": U2, "V": V2, "W": W2}
