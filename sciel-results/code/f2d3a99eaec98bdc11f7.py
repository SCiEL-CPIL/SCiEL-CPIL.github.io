import numpy as np
import time as _time

_DEADLINE = 130.0

def construct():
    _t0 = _time.monotonic()
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    U_tbl = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    V_tbl = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    W_tbl = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for a in range(2):
        j_off = a * 2
        for b in range(3):
            k_off = b * 2
            for s in range(7):
                col = a * 21 + b * 7 + s
                if col >= rank:
                    continue
                if _time.monotonic() - _t0 > _DEADLINE:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                for i_loc, j_loc, c in U_tbl[s]:
                    U[i_loc * m + (j_off + j_loc), col] += c
                for j_loc, k_loc, c in V_tbl[s]:
                    V[(j_off + j_loc) * p + (k_off + k_loc), col] += c
                for i_loc, k_loc, c in W_tbl[s]:
                    W[(k_off + k_loc) * n + i_loc, col] += c
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
