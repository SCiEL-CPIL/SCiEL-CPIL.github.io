import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    A_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    B_coeffs = [
        [1, 1, 1, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    C_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 1]   # M7
    ]
    
    for t in range(4):
        for s in range(7):
            col = 7 * t + s
            a_vec = A_coeffs[s]
            b_vec = B_coeffs[s]
            c_vec = C_coeffs[s]
            
            # Fill U
            for u_idx in range(4):
                U[u_idx][col] = a_vec[u_idx]
                
            # Fill V
            # v_global = 8*j + 2*t + delta
            # v_local = 2*j + delta
            for v_glob in range(16):
                j = v_glob // 8
                k_glob = v_glob % 8
                t_check = k_glob // 2
                if t_check == t:
                    delta = k_glob % 2
                    v_loc = 2 * j + delta
                    V[v_glob][col] = b_vec[v_loc]
                    
            # Fill W
            # w_global = 16*t + 8*delta + i
            # w_local = 2*delta + i
            for w_glob in range(16):
                t_check = w_glob // 16
                if t_check == t:
                    rem = w_glob % 16
                    delta = rem // 8
                    i = rem % 8
                    w_loc = 2 * delta + i
                    W[w_glob][col] = c_vec[w_loc]
                    
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
