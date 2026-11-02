import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    for mat in (U_list, V_list, W_list):
        for row in mat:
            for idx, val in enumerate(row):
                if val == 1.0:
                    row[idx] = 1
                elif val == 0.0:
                    row[idx] = 0

    return {"U": U_list, "V": V_list, "W": W_list}
