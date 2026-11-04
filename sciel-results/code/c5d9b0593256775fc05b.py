import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = 2, 5, 6
    
    # Total rank is 54
    rank = 54
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 multiplication
    # U, V, W are 2x2 matrices, flattened row-major
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 1, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 1, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 1]
    ]
    
    col = 0
    
    # Helper to add a Strassen block for j_start in {0, 2} and k_start in {0, 2, 4}
    def add_strassen_block(j_start, k_start):
        nonlocal col
        for r in range(7):
            # U: 2x2 block at (i in {0,1}, j in {j_start, j_start+1})
            for i in range(2):
                for jl in range(2):
                    j = j_start + jl
                    U[i * m + j, col] = strassen_U[r][i * 2 + jl]
            
            # V: 2x2 block at (j in {j_start, j_start+1}, k in {k_start, k_start+1})
            for jl in range(2):
                for kl in range(2):
                    j = j_start + jl
                    k = k_start + kl
                    V[j * p + k, col] = strassen_V[r][jl * 2 + kl]
            
            # W: 2x2 block at (k in {k_start, k_start+1}, i in {0,1})
            for kl in range(2):
                for i in range(2):
                    k = k_start + kl
                    W[k * n + i, col] = strassen_W[r][kl * 2 + i]
            
            col += 1
    
    # Block 1: j in {0, 1}
    for k_start in [0, 2, 4]:
        add_strassen_block(0, k_start)
    
    # Block 2: j in {2, 3}
    for k_start in [0, 2, 4]:
        add_strassen_block(2, k_start)
    
    # Block 3: j in {4}
    # Schoolbook for the remaining 1x6 part
    # We need U[i, 4] V[4, k] W[k, i] = 1 for all i, k
    # We use 12 terms, one for each (i, k)
    for i in range(n):
        for k in range(p):
            U[i * m + 4, col] = 1.0
            V[4 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Verify rank
    if col != rank:
        # Fallback to schoolbook if construction fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
