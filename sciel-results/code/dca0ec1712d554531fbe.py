import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = n * m * p  # 18
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[a_idx(i, j), t] = 1.0
                V[b_idx(j, k), t] = 1.0
                W[w_idx(k, i), t] = 1.0
                t += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
