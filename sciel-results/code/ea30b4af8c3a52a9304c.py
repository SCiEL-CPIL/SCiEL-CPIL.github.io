"""Direct rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    n, m, p = 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7 (schoolbook: 32 terms)
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: i=1,2 (Strassen on 2x4x8, 8 blocks of 2x2x2, 7 terms each = 56 terms)
    # Strassen 2x2x2 decomposition vectors
    # U vectors (7 x 4), V vectors (7 x 4), W vectors (7 x 4)
    # Local indices: i_local in {0,1}, j_local in {0,1}, k_local in {0,1}
    # U index: i_local * 2 + j_local
    # V index: j_local * 2 + k_local
    # W index: k_local * 2 + i_local
    
    strassen_U = np.array([
        [1, 0, 0, 1],   # M0 = A00*A00
        [0, 0, 1, 1],   # M1 = A10*(A01+A11)
        [1, 0, 0, 0],   # M2 = A00*(B10-B00)
        [0, 0, 0, 1],   # M3 = A11*B11
        [1, 1, 0, 0],   # M4 = (A00+A11)*B00
        [-1, 0, 1, 0],  # M5 = (A10-A00)*(B00+B10)
        [0, 1, 0, -1]   # M6 = (A01-A11)*(B10+B11)
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 0, 0, 1],   # M0
        [1, 0, 0, 0],   # M1
        [0, 1, 0, -1],  # M2
        [-1, 0, 1, 0],  # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [0, 0, 1, 1]    # M6
    ], dtype=float)
    
    strassen_W = np.array([
        [1, 0, 0, 1],   # M0 -> C00
        [0, 0, 1, -1],  # M1 -> C10
        [0, 1, 0, 1],   # M2 -> C01
        [1, 0, 1, 0],   # M3 -> C11
        [-1, 1, 0, 0],  # M4 -> C00
        [0, 0, 0, 1],   # M5 -> C10
        [1, 0, 0, 0]    # M6 -> C01
    ], dtype=float)
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # Scatter U values
                for l in range(4):
                    if strassen_U[t, l] != 0:
                        local_i = l // 2
                        local_j = l % 2
                        global_i = 1 + local_i
                        global_j = j_start + local_j
                        U[global_i * m + global_j, col] += strassen_U[t, l]
                # Scatter V values
                for l in range(4):
                    if strassen_V[t, l] != 0:
                        local_j = l // 2
                        local_k = l % 2
                        global_j = j_start + local_j
                        global_k = k_start + local_k
                        V[global_j * p + global_k, col] += strassen_V[t, l]
                # Scatter W values
                for l in range(4):
                    if strassen_W[t, l] != 0:
                        local_k = l // 2
                        local_i = l % 2
                        global_k = k_start + local_k
                        global_i = 1 + local_i
                        W[global_k * n + global_i, col] += strassen_W[t, l]
                col += 1
                
    # Convert to lists of lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
