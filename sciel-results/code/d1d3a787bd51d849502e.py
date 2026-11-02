import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin from 120s limit
    
    # Dimensions
    n, m, p = 2, 2, 9
    
    # Rank 32 decomposition
    rank = 32
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term
    def add_term(idx, u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx, idx] += u_val
        V[v_idx, idx] += v_val
        W[w_idx, idx] += w_val
        
    # Strassen blocks for k=0..7
    for block_idx in range(4):
        if time.time() > deadline:
            break
            
        k0 = 2 * block_idx
        k1 = k0 + 1
        base_term = block_idx * 7
        
        # B indices
        b_e = k0       # B[0, k0]
        b_f = k1       # B[0, k1]
        b_g = 9 + k0   # B[1, k0]
        b_h = 9 + k1   # B[1, k1]
        
        # C indices (W indices)
        # W index = k*n + i = 2*k + i
        c_00 = 2 * k0     # C[0, k0] -> i=0, k=k0
        c_01 = 2 * k1     # C[0, k1] -> i=0, k=k1
        c_10 = 2 * k0 + 1 # C[1, k0] -> i=1, k=k0
        c_11 = 2 * k1 + 1 # C[1, k1] -> i=1, k=k1
        
        # A indices
        a_00 = 0
        a_01 = 1
        a_10 = 2
        a_11 = 3
        
        # M1 = (a00 + a11) * (b_e + b_h)
        # C00 += M1, C11 += M1
        add_term(base_term + 0, a_00, 1, b_e, 1, c_00, 1)
        add_term(base_term + 0, a_11, 1, b_h, 1, c_11, 1)
        
        # M2 = (a10 + a11) * b_e
        # C10 += M2, C11 -= M2
        add_term(base_term + 1, a_10, 1, b_e, 1, c_10, 1)
        add_term(base_term + 1, a_11, 1, b_e, 1, c_11, -1)
        
        # M3 = a00 * (b_f - b_h)
        # C01 += M3, C11 += M3
        add_term(base_term + 2, a_00, 1, b_f, 1, c_01, 1)
        add_term(base_term + 2, a_00, 1, b_h, -1, c_11, 1)
        
        # M4 = a11 * (b_g - b_e)
        # C00 -= M4, C10 += M4
        add_term(base_term + 3, a_11, 1, b_g, 1, c_00, -1)
        add_term(base_term + 3, a_11, 1, b_e, -1, c_10, 1)
        
        # M5 = (a00 + a01) * b_h
        # C00 -= M5, C01 += M5
        add_term(base_term + 4, a_00, 1, b_h, 1, c_00, -1)
        add_term(base_term + 4, a_01, 1, b_h, 1, c_01, 1)
        
        # M6 = (a10 - a00) * (b_e + b_f)
        # C11 += M6
        add_term(base_term + 5, a_10, 1, b_e, 1, c_11, 1)
        add_term(base_term + 5, a_00, -1, b_e, 1, c_11, 1)
        add_term(base_term + 5, a_10, 1, b_f, 1, c_11, 1)
        add_term(base_term + 5, a_00, -1, b_f, 1, c_11, 1)
        
        # M7 = (a01 - a11) * (b_g + b_h)
        # C00 += M7
        add_term(base_term + 6, a_01, 1, b_g, 1, c_00, 1)
        add_term(base_term + 6, a_11, -1, b_g, 1, c_00, 1)
        add_term(base_term + 6, a_01, 1, b_h, 1, c_00, 1)
        add_term(base_term + 6, a_11, -1, b_h, 1, c_00, 1)

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Schoolbook for column 8
    k = 8
    base_term = 28
    
    # C[0,8] = A[0,0]B[0,8] + A[0,1]B[1,8]
    # W index for C[0,8] is 2*8 + 0 = 16
    # W index for C[1,8] is 2*8 + 1 = 17
    
    add_term(base_term + 0, 0, 1, k, 1, 16, 1)
    add_term(base_term + 1, 1, 1, 9+k, 1, 16, 1)
    add_term(base_term + 2, 2, 1, k, 1, 17, 1)
    add_term(base_term + 3, 3, 1, 9+k, 1, 17, 1)

    return {"U": U, "V": V, "W": W}
