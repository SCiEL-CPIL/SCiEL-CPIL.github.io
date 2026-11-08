"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    U_dim = n * m  # 16
    V_dim = m * p  # 28
    W_dim = p * n  # 28

    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((U_dim, rank), dtype=np.int64)
    V = np.zeros((V_dim, rank), dtype=np.int64)
    W = np.zeros((W_dim, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    def to_list(arr):
        return arr.astype(int).tolist()

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
