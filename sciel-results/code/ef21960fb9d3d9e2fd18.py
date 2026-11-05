import numpy as np
import time

T0 = time.time()
DEADLINE = 145.0

def construct():
    n, m, p = 4, 4, 7

    # Strassen 2x2x2: (U_sp, V_sp, W_sp) for each of 7 terms
    S22 = [
        (([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])),
        (([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])),
        (([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])),
        (([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])),
        (([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])),
        (([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])),
        (([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])),
    ]

    # Strassen 4x4x4 outer block selections (row-blocks for A, col-blocks for B)
    S44_X = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    S44_Y = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    S44_C = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]

    RANK_TOTAL = 93
    U = np.zeros((n * m, RANK_TOTAL), dtype=np.float64)
    V = np.zeros((m * p, RANK_TOTAL), dtype=np.float64)
    W = np.zeros((p * n, RANK_TOTAL), dtype=np.float64)

    col = 0

    # --- Part 1: 4x4x4 via Strassen (rank 49 = 7*7) ---
    for t in range(7):
        xsel = S44_X[t]
        ysel = S44_Y[t]
        cblk = S44_C[t]
        for s in range(7):
            r = col
            col += 1
            u_pts, v_pts, w_pts = S22[s]
            for (si, sj, sv) in u_pts:
                for (blk, bv) in xsel.items():
                    ii = 2 * (blk // 2) + si
                    jj = 2 * (blk % 2) + sj
                    U[ii * m + jj, r] += sv * bv
            for (si, sj, sv) in v_pts:
                for (blk, bv) in ysel.items():
                    ii = 2 * (blk // 2) + si
                    jj = 2 * (blk % 2) + sj
                    V[ii * p + jj, r] += sv * bv
            for (si, sj, sv) in w_pts:
                if sv == 0:
                    continue
                for ((bi, bj), cv) in cblk.items():
                    ii = 2 * bi + si
                    kk = 2 * bj + sj
                    W[kk * n + ii, r] += sv * cv

    # --- Part 2: 4x4x3 as four 2x2x3 blocks (rank 44 = 4*11) ---
    def fill_223(arow, acol, brow, crw):
        nonlocal col
        # 7 Strassen 2x2x2 terms for columns 4,5 of output
        for s in range(7):
            r = col
            col += 1
            u_pts, v_pts, w_pts = S22[s]
            for (si, sj, sv) in u_pts:
                U[(2*arow + si) * m + (2*acol + sj), r] += sv
            for (si, sj, sv) in v_pts:
                V[(2*brow + si) * p + (4 + sj), r] += sv
            for (si, sj, sv) in w_pts:
                if sv != 0:
                    W[(4 + sj) * n + (2*crw + si), r] += sv
        # 4 schoolbook terms for column 6
        for si in range(2):
            for sj in range(2):
                r = col
                col += 1
                U[(2*arow + si) * m + (2*acol + sj), r] = 1.0
                V[(2*brow + si) * p + 6, r] = 1.0
                W[6 * n + (2*crw + si), r] = 1.0

    fill_223(0, 0, 0, 0)
    fill_223(0, 1, 1, 0)
    fill_223(1, 0, 0, 1)
    fill_223(1, 1, 1, 1)

    assert col == RANK_TOTAL, "rank mismatch: %d" % col

    # Verify quickly if time permits
    if time.time() - T0 < DEADLINE - 3.0:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(RANK_TOTAL):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    e = abs(s - 1.0)
                    if e > max_err:
                        max_err = e

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
