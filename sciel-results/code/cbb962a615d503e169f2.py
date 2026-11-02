import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nrm = n * m
    mp = m * p
    pn = p * n
    nblocks = p // 2
    perblock = 7
    total_rank = nblocks * perblock

    t0 = time.perf_counter()
    hard_limit = 110.0

    U_arr = np.zeros((nrm, total_rank), dtype=np.int32)
    V_arr = np.zeros((mp, total_rank), dtype=np.int32)
    W_arr = np.zeros((pn, total_rank), dtype=np.int32)

    su = (
        (1, 0, 0, 1),
        (0, 0, 1, 1),
        (1, 0, 0, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (-1, 0, 1, 0),
        (0, 1, 0, -1),
    )
    sv = (
        (1, 0, 0, 1),
        (1, 0, 0, 0),
        (0, 1, 0, -1),
        (-1, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (0, 0, 1, 1),
    )
    sw = (
        (1, 0, 0, 1),
        (0, 0, 1, -1),
        (0, 1, 0, 1),
        (1, 0, 1, 0),
        (-1, 1, 0, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 0),
    )

    for blk in range(nblocks):
        if time.perf_counter() - t0 > hard_limit:
            break
        c0 = blk * perblock
        rowV = (2 * blk, 2 * blk + 1, mp - 2 * (nblocks - blk), mp - 2 * (nblocks - blk) + 1)
        rowW = (4 * blk, 4 * blk + 2, 4 * blk + 1, 4 * blk + 3)
        for stp in range(perblock):
            cc = c0 + stp
            for ax in range(4):
                if su[stp][ax]:
                    U_arr[ax, cc] = su[stp][ax]
                if sv[stp][ax]:
                    V_arr[rowV[ax], cc] = sv[stp][ax]
                if sw[stp][ax]:
                    W_arr[rowW[ax], cc] = sw[stp][ax]

    return {
        "U": U_arr.tolist(),
        "V": V_arr.tolist(),
        "W": W_arr.tolist(),
    }
# EVOLVE-BLOCK-END
