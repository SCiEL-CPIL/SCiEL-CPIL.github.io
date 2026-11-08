import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    rank = 49 + 80
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook for k=4..8
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    def get_block_indices_A(br, bc):
        base_r = br * 2
        base_c = bc * 2
        return [
            (base_r + 0) * 4 + (base_c + 0),
            (base_r + 0) * 4 + (base_c + 1),
            (base_r + 1) * 4 + (base_c + 0),
            (base_r + 1) * 4 + (base_c + 1),
        ]
    
    def get_block_indices_B(br, bc):
        base_r = br * 2
        base_c = bc * 2
        return [
            (base_r + 0) * 9 + (base_c + 0),
            (base_r + 0) * 9 + (base_c + 1),
            (base_r + 1) * 9 + (base_c + 0),
            (base_r + 1) * 9 + (base_c + 1),
        ]
    
    def get_block_indices_C(br, bc):
        base_k = br * 2
        base_i = bc * 2
        return [
            (base_k + 0) * 4 + (base_i + 0),
            (base_k + 0) * 4 + (base_i + 1),
            (base_k + 1) * 4 + (base_i + 0),
            (base_k + 1) * 4 + (base_i + 1),
        ]
    
    def add_term(u_entries, v_entries, w_entries, col_idx):
        for (idx, val) in u_entries:
            U[idx][col_idx] += val
        for (idx, val) in v_entries:
            V[idx][col_idx] += val
        for (idx, val) in w_entries:
            W[idx][col_idx] += val
    
    high_level_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    strassen_2x2 = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]),
    ]
    
    C_block_contributions = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1, 3: -1},
    ]
    
    col = 0
    for H_idx, (A_coeffs, B_coeffs) in enumerate(high_level_terms):
        for s_idx, (X_idx, Y_idx, Z_idx) in enumerate(strassen_2x2):
            u_entries = []
            for (local_pos, s_coeff) in X_idx:
                for (br, bc) in A_coeffs:
                    a_coeff = A_coeffs[(br,bc)]
                    global_idx = get_block_indices_A(br, bc)[local_pos]
                    u_entries.append((global_idx, a_coeff * s_coeff))
            
            v_entries = []
            for (local_pos, s_coeff) in Y_idx:
                for (br, bc) in B_coeffs:
                    b_coeff = B_coeffs[(br,bc)]
                    global_idx = get_block_indices_B(br, bc)[local_pos]
                    v_entries.append((global_idx, b_coeff * s_coeff))
            
            w_entries = []
            for (local_pos, s_coeff) in Z_idx:
                for (c_block, c_coeff) in C_block_contributions[H_idx].items():
                    br = c_block // 2
                    bc = c_block % 2
                    global_idx = get_block_indices_C(br, bc)[local_pos]
                    w_entries.append((global_idx, c_coeff * s_coeff))
            
            add_term(u_entries, v_entries, w_entries, col)
            col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
