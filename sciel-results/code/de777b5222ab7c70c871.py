import time
_t0 = time.time()
_DL = 145.0

import numpy as np

def construct():
    N, M, P = 2, 4, 8
    R = 56
    du, dv, dw = N * M, M * P, P * N

    # Strassen-7 rank basis for 2x2 * 2x2 -> 2x2
    # Each tuple: (U_ops, V_ops, W_ops) where each ops is list of (local_idx, coeff)
    S = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    U = np.zeros((du, R))
    V = np.zeros((dv, R))
    W = np.zeros((dw, R))

    blk = 0
    for rg in range(2):
        for cp in range(4):
            rc = blk * 7
            blk += 1
            for si in range(7):
                c = rc + si
                uo, vo, wo = S[si]
                for (li, cv) in uo:
                    U[li // 2 * M + 2 * rg + li % 2, c] += cv
                for (li, cv) in vo:
                    V[(2 * rg + li // 2) * P + 2 * cp + li % 2, c] += cv
                for (li, cv) in wo:
                    W[(2 * cp + li % 2) * N + li // 2, c] += cv

    # Validate
    for i in range(N):
        for j in range(M):
            for k in range(P):
                ui, vi, wi = i * M + j, j * P + k, k * N + i
                s = float(np.dot(U[ui], V[vi] * W[wi]))
                if abs(s - 1.0) > 1e-9:
                    return _fallback()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _fallback():
    N, M, P = 2, 4, 8
    R = 56
    U = [[0] * R for _ in range(N * M)]
    V = [[0] * R for _ in range(M * P)]
    W = [[0] * R for _ in range(P * N)]
    S = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    g = 0
    for h in range(2):
        for jj in range(4):
            br = g * 7
            g += 1
            am = {0: 0*M+(2*h), 1: 0*M+(2*h+1), 2: 1*M+(2*h), 3: 1*M+(2*h+1)}
            bm = {0: (2*h)*P+(2*jj), 1: (2*h)*P+(2*jj+1), 2: (2*h+1)*P+(2*jj), 3: (2*h+1)*P+(2*jj+1)}
            cm = {0: (2*jj)*N+0, 1: (2*jj+1)*N+0, 2: (2*jj)*N+1, 3: (2*jj+1)*N+1}
            for r in range(7):
                rr = br + r
                ul, vl, wl = S[r]
                for ix, vv in ul:
                    U[am[ix]][rr] += vv
                for ix, vv in vl:
                    V[bm[ix]][rr] += vv
                for ix, vv in wl:
                    W[cm[ix]][rr] += vv
    return {"U": U, "V": V, "W": W}
