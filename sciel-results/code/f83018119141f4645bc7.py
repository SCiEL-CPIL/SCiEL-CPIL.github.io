import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100
    
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_term(u_vals, v_vals, w_vals):
        terms.append((u_vals, v_vals, w_vals))
        
    def strassen_222(i_off, j_off, k_off):
        # M1
        add_term({(i_off, j_off): 1, (i_off+1, j_off+1): 1},
                 {(j_off, k_off): 1, (j_off+1, k_off+1): 1},
                 {(k_off, i_off): 1, (k_off+1, i_off+1): 1})
        # M2
        add_term({(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
                 {(j_off, k_off): 1},
                 {(k_off, i_off+1): 1})
        # M3
        add_term({(i_off, j_off): 1},
                 {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1},
                 {(k_off+1, i_off): 1, (k_off+1, i_off+1): 1})
        # M4
        add_term({(i_off+1, j_off+1): 1},
                 {(j_off, k_off+1): 1, (j_off, k_off): -1},
                 {(k_off, i_off): 1, (k_off, i_off+1): 1})
        # M5
        add_term({(i_off, j_off): 1, (i_off, j_off+1): 1},
                 {(j_off+1, k_off+1): 1},
                 {(k_off+1, i_off): 1})
        # M6
        add_term({(i_off+1, j_off): 1, (i_off, j_off): -1},
                 {(j_off, k_off): 1, (j_off, k_off+1): 1},
                 {(k_off, i_off+1): 1, (k_off+1, i_off+1): 1})
        # M7
        add_term({(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
                 {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
                 {(k_off, i_off): 1, (k_off+1, i_off): 1})

    def schoolbook(i, j, k):
        add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    # 1. Core 4x4x4 (j in 0..3, k in 0..3)
    # Split into four 2x2x2 blocks
    strassen_222(0, 0, 0) # i:0-1, j:0-1, k:0-1
    strassen_222(0, 0, 2) # i:0-1, j:0-1, k:2-3
    strassen_222(0, 2, 0) # i:0-1, j:2-3, k:0-1
    strassen_222(0, 2, 2) # i:0-1, j:2-3, k:2-3
    
    strassen_222(2, 0, 0) # i:2-3, j:0-1, k:0-1
    strassen_222(2, 0, 2) # i:2-3, j:0-1, k:2-3
    strassen_222(2, 2, 0) # i:2-3, j:2-3, k:0-1
    strassen_222(2, 2, 2) # i:2-3, j:2-3, k:2-3
    
    # 2. j=4 slice (i in 0..3, j=4, k in 0..4)
    for i in range(n):
        for k in range(p):
            schoolbook(i, 4, k)
            
    # 3. k=4 slice (i in 0..3, j in 0..3, k=4)
    # Note: j=4, k=4 is already covered in step 2.
    for i in range(n):
        for j in range(4): # j in 0..3
            schoolbook(i, j, 4)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (u_vals, v_vals, w_vals) in enumerate(terms):
        for (i, j), val in u_vals.items():
            U[i * m + j][r] = val
        for (j, k), val in v_vals.items():
            V[j * p + k][r] = val
        for (k, i), val in w_vals.items():
            W[k * n + i][r] = val

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = float(np.dot(U_arr[i*m+j], np.multiply(V_arr[jp*p+k], W_arr[k*n+i])))
                    expected = 1.0 if j == jp else 0.0
                    err = abs(val - expected)
                    if err > max_err:
                        max_err = err

    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
