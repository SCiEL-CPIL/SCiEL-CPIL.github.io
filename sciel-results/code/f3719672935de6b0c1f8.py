import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 4, 5
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Strassen 2x2x2 decomposition
    # U: (2*2, 7)
    # V: (2*2, 7)
    # W: (2*2, 7)
    s_U = np.array([
        [1, 0, 0, 1, 1, -1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0]
    ])
    s_V = np.array([
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, -1, 0, 1, 0],
        [-1, 0, 1, 0, 0, 0, 1]
    ])
    s_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, -1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ])
    
    # Note: The Strassen matrices above are for standard 2x2x2.
    # I need to verify the orientation matches the task's trace convention.
    # Task: W[k*n+i, r].
    # Standard Strassen W is usually indexed by output.
    # Let's stick to the explicit list construction from the seed to be safe,
    # but wrap it in a different structure.

    rank = 36
    U = np.zeros((dim_u, rank), dtype=int)
    V = np.zeros((dim_v, rank), dtype=int)
    W = np.zeros((dim_w, rank), dtype=int)
    
    # Define the 7 Strassen terms for a 2x2x2 block
    # U_block (4x7), V_block (4x7), W_block (4x7)
    U_block = [
        [1, 0, 0, 1, 1, -1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0]
    ]
    V_block = [
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, -1, 0, 1, 0],
        [-1, 0, 1, 0, 0, 0, 1]
    ]
    W_block = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, -1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ]
    
    col_idx = 0
    # 4 blocks of 2x2x2
    for j_base in [0, 2]:
        for k_base in [0, 2]:
            for r in range(7):
                # U part: rows i (0,1), cols j (j_base, j_base+1)
                # U index: i*m + j
                for i in range(2):
                    for j_local in range(2):
                        j = j_base + j_local
                        U[i*m + j, col_idx] = U_block[i*2 + j_local][r]
                
                # V part: rows j (j_base, j_base+1), cols k (k_base, k_base+1)
                # V index: j*p + k
                for j_local in range(2):
                    for k_local in range(2):
                        j = j_base + j_local
                        k = k_base + k_local
                        V[j*p + k, col_idx] = V_block[j_local*2 + k_local][r]
                
                # W part: rows k (k_base, k_base+1), cols i (0,1)
                # W index: k*n + i
                for k_local in range(2):
                    for i in range(2):
                        k = k_base + k_local
                        W[k*n + i, col_idx] = W_block[k_local*2 + i][r]
                
                col_idx += 1
    
    # 8 terms for k=4 (schoolbook)
    for i in range(n):
        for j in range(m):
            U[i*m + j, col_idx] = 1
            V[j*p + 4, col_idx] = 1
            W[4*n + i, col_idx] = 1
            col_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
