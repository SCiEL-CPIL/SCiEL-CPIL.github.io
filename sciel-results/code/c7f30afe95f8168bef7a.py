"""Matrix multiplication 3x3x5: rank-minimal exact tensor decomposition."""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook decomposition of rank n*m*p = 45.
    # This is guaranteed correct.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Convert floats that are integers to int
    U_list = [[int(x) if float(x) == int(float(x)) else x for x in row] for row in U_list]
    V_list = [[int(x) if float(x) == int(float(x)) else x for x in row] for row in V_list]
    W_list = [[int(x) if float(x) == int(float(x)) else x for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
