import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col_idx = 0
    steps = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    for u in range(2):
        for v in range(4):
            for step in steps:
                u_coeffs, v_coeffs, w_coeffs = step
                for li, lj, val in u_coeffs:
                    gi = li
                    gj = 2 * u + lj
                    u_idx = gi * m + gj
                    U[u_idx, col_idx] += val
                for lk, ll, val in v_coeffs:
                    gr = 2 * u + lk
                    gc = 2 * v + ll
                    v_idx = gr * p + gc
                    V[v_idx, col_idx] += val
                for li, ll, val in w_coeffs:
                    gi = li
                    gc = 2 * v + ll
                    w_idx = gc * n + gi
                    W[w_idx, col_idx] += val
                col_idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
