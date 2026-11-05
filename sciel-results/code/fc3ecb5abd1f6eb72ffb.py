import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n  # 6

    # Schoolbook decomposition: rank = n*m*p = 216
    total_rank = dim * dim * dim

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    col_idx = 0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                u_idx = i * dim + j
                v_idx = j * dim + k
                w_idx = k * dim + i
                U[u_idx, col_idx] = 1.0
                V[v_idx, col_idx] = 1.0
                W[w_idx, col_idx] = 1.0
                col_idx += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
