import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 36 for schoolbook
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[int(x) if x == int(x) else float(x) for x in row] for row in U]
    V_list = [[int(x) if x == int(x) else float(x) for x in row] for row in V]
    W_list = [[int(x) if x == int(x) else float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
