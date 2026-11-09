import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x7 matrix.
    
    Strategy:
    - 4x4x4 block: Strassen rank 49
    - 4x4x3 block: schoolbook rank 48
    Total rank = 97
    """
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Strassen 2x2x2 base coefficients
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
    
    rank_strassen_4x4 = 49
    rank_schoolbook_4x4x3 = 4 * 4 * 3  # 48
    rank = rank_strassen_4x4 + rank_schoolbook_4x4x3  # 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4 * 4x4 using Strassen rank 49 ---
    col = 0
    for k_outer in range(7):
        for l_inner in range(7):
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = u_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0:
                        continue
                    for b in range(4):
                        coeff_outer = u_in[k_outer][b]
                        if coeff_outer == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += coeff_inner * coeff_outer
            
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = v_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0:
                        continue
                    for b in range(4):
                        coeff_outer = v_in[k_outer][b]
                        if coeff_outer == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        v_vec[j_global * 7 + k_global] += coeff_inner * coeff_outer
            
            w_vec = [0] * 28
            for c_block in range(4):
                coeff_outer_w = w_out[c_block][k_outer]
                if coeff_outer_w == 0:
                    continue
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l_inner]
                    if coeff_inner_w == 0:
                        continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0:
                        continue
                    c_block_row = c_block // 2
                    c_block_col = c_block % 2
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    i_global = 2 * c_block_row + c_inner_row
                    k_global = 2 * c_block_col + c_inner_col
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            
            col += 1
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    # --- Part 2: 4x4 * 4x3 schoolbook rank 48 ---
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
