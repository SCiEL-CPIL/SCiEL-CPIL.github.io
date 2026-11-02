import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 130.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for j1 in range(2):
        # Top part: 2x2x8
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                # U
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j, col] += coeff
                
                # V
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k, col] += coeff
                
                # W
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
        
        # Bot part: 1x2x8
        for k in range(8):
            for j2 in range(2):
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j, col] = 1
                V[global_j * p + k, col] = 1
                W[k * n + global_i, col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
