import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    rank = 97
    
    # Use NumPy for faster construction
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # --- Part 1: 4x4 * 4x4 (Left part of B) ---
    # Strassen 4x4x4 (Rank 49)
    
    u_in = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    u_out = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_out = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_outer = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    for k in range(7):
        for l in range(7):
            # U vector
            u_vec = np.zeros(16, dtype=int)
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = u_in[l][i_inner * 2 + j_inner]
                    if coeff_inner == 0: continue
                    for b in range(4):
                        coeff_outer = u_out[k][b]
                        if coeff_outer == 0: continue
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += coeff_inner * coeff_outer
            
            # V vector
            v_vec = np.zeros(28, dtype=int)
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = v_in[l][i_inner * 2 + j_inner]
                    if coeff_inner == 0: continue
                    for b in range(4):
                        coeff_outer = v_out[k][b]
                        if coeff_outer == 0: continue
                        b_row = b // 2
                        b_col = b % 2
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        v_vec[j_global * 7 + k_global] += coeff_inner * coeff_outer
            
            # W vector
            w_vec = np.zeros(28, dtype=int)
            for c_block in range(4):
                coeff_outer_w = w_outer[c_block][k]
                if coeff_outer_w == 0: continue
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l]
                    if coeff_inner_w == 0: continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0: continue
                    
                    c_block_row = c_block // 2
                    c_block_col = c_block % 2
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    
                    i_global = 2 * c_block_row + c_inner_row
                    k_global = 2 * c_block_col + c_inner_col
                    
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # --- Part 2: 4x4 * 4x3 (Right part of B) ---
    # Schoolbook Rank 48
    if time.time() < deadline:
        for i in range(4):
            for j in range(4):
                for k_prime in range(3):
                    if time.time() > deadline:
                        break
                    k_global = 4 + k_prime
                    U[i * 4 + j, col] = 1
                    V[j * 7 + k_global, col] = 1
                    W[k_global * 4 + i, col] = 1
                    col += 1
    
    # Convert to lists for output
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
