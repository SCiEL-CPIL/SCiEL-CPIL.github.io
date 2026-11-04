import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    deadline = start_time + 110  # Safety margin well before 120s limit
    
    # Block structure: 6x6 = 2x2 blocks of 3x3
    block_size = 3
    
    total_rank = 189
    
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    # Strassen coefficients precomputed as lists for speed
    # Each entry: (list_of_X, list_of_Y, list_of_C)
    # X: [(ib, jb, coeff), ...]
    # Y: [(jb, kb, coeff), ...]
    # C: [(ib, kb, coeff), ...]
    precomputed_steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10 + A11) * B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00 * (B01 - B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11 * (B10 - B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00 + A01) * B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    bs = block_size
    dim = n  # 6
    
    for x_list, y_list, c_list in precomputed_steps:
        for r in range(bs):
            for s in range(bs):
                for c in range(bs):
                    # U part
                    for (ib, jb, x_coeff) in x_list:
                        u_idx = (ib * bs + r) * dim + (jb * bs + s)
                        U[u_idx, col] += x_coeff
                    
                    # V part
                    for (jb, kb, y_coeff) in y_list:
                        v_idx = (jb * bs + s) * dim + (kb * bs + c)
                        V[v_idx, col] += y_coeff
                    
                    # W part (trace convention: W[k*n+i, r])
                    for (ib, kb, c_coeff) in c_list:
                        w_idx = (kb * bs + c) * dim + (ib * bs + r)
                        W[w_idx, col] += c_coeff
                    
                    col += 1
                    
                    # Periodic time check every 27 iterations (one full Strassen step)
                    if col % 27 == 0 and time.time() > deadline:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
