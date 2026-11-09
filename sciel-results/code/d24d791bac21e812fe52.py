import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _strassen_2x2x2():
    s = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    return s

def _schoolbook_1x2x2():
    s = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1}, {(1,0):1}),
        ({(0,1):1}, {(1,1):1}, {(1,0):1}),
    ]
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = 3, 4, 4
    strassen = _strassen_2x2x2()
    schoolbook = _schoolbook_1x2x2()
    total_rank = 4 * len(strassen) + 4 * len(schoolbook)  # 28 + 16 = 44
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    rc = 0
    def _emit(i_off, j_off, k_off, terms):
        nonlocal rc
        for (ud, vd, wd) in terms:
            for (li, lj), c in ud.items():
                U[(i_off + li) * m + (j_off + lj), rc] += c
            for (lj, lk), c in vd.items():
                V[(j_off + lj) * p + (k_off + lk), rc] += c
            for (lk, li), c in wd.items():
                W[(k_off + lk) * n + (i_off + li), rc] += c
            rc += 1
    _emit(0, 0, 0, strassen)
    _emit(0, 2, 0, strassen)
    _emit(0, 0, 2, strassen)
    _emit(0, 2, 2, strassen)
    if time.time() > deadline - 5:
        return {"U": U[:, :rc].tolist(), "V": V[:, :rc].tolist(), "W": W[:, :rc].tolist()}
    _emit(2, 0, 0, schoolbook)
    _emit(2, 2, 0, schoolbook)
    _emit(2, 0, 2, schoolbook)
    _emit(2, 2, 2, schoolbook)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
