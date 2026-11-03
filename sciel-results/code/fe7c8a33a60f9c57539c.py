import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 6, 6, 6
    bs = 3
    nb = n // bs
    rank = nb ** 3 * 7

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2 coefficients
    strassen_terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                for term in strassen_terms:
                    u_pos, v_pos, w_pos = term
                    for (r, c, sgn) in u_pos:
                        idx = (ib * bs + r) * m + (kb * bs + c)
                        U[idx, col] += sgn
                    for (r, c, sgn) in v_pos:
                        idx = (kb * bs + r) * p + (jb * bs + c)
                        V[idx, col] += sgn
                    for (r, c, sgn) in w_pos:
                        idx = (jb * bs + c) * n + (ib * bs + r)
                        W[idx, col] += sgn
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
