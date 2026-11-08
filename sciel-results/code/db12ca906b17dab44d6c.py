"""
Rank 44 tensor decomposition for 2x3x8 matrix multiplication.
Uses Strassen (rank 7) for 2x2x2 blocks and outer product (rank 4) for 2x1x2 blocks.
4 blocks of 2 columns in B: 4*(7+4) = 44.
"""
import time

_START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            if c != 0:
                U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            if c != 0:
                V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            if c != 0:
                W[idx][col] += c

    current_col = 0
    
    for k_block in range(4):
        if time.time() - _START_TIME > TIME_LIMIT:
            break
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        # Strassen 2x2x2 for j=0,1
        a_idx, b_idx, c_idx, d_idx = 0, 1, 3, 4
        e_idx, f_idx = 0 * p + k0, 0 * p + k1
        g_idx, h_idx = 1 * p + k0, 1 * p + k1
        w_00, w_01 = k0 * n, k1 * n
        w_10, w_11 = k0 * n + 1, k1 * n + 1
        
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Outer product 2x1x2 for j=2
        a2_idx, c2_idx = 2, 5
        e2_idx, f2_idx = 2 * p + k0, 2 * p + k1
        
        add_term(current_col, [a2_idx], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        add_term(current_col, [a2_idx], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        add_term(current_col, [c2_idx], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        add_term(current_col, [c2_idx], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}
