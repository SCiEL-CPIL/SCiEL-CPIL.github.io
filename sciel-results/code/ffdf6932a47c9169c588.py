import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen vectors for 2x2x2
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11
    U_in = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, -1, 0, 1] # M7: A01-A11
    ]
    V_in = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    W_in = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11
        [0, 1, 0, 1], # M3: C01+C11
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 1, 0, 0],# M5: -C00+C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Outer Strassen block combinations
    # Blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # U_out[i] is a list of (block_idx, coeff)
    U_out = [
        [(0, 1), (3, 1)], # M1: A00+A11
        [(2, 1), (3, 1)], # M2: A10+A11
        [(0, 1)],         # M3: A00
        [(3, 1)],         # M4: A11
        [(0, 1), (1, 1)], # M5: A00+A01
        [(2, 1), (0, -1)],# M6: A10-A00
        [(1, 1), (3, -1)] # M7: A01-A11
    ]
    V_out = [
        [(0, 1), (3, 1)], # M1: B00+B11
        [(0, 1)],         # M2: B00
        [(1, 1), (3, -1)],# M3: B01-B11
        [(2, 1), (0, -1)],# M4: B10-B00
        [(3, 1)],         # M5: B11
        [(0, 1), (1, 1)], # M6: B00+B01
        [(2, 1), (3, 1)]  # M7: B10+B11
    ]
    # W_out[i] is a dict {C_block_idx: coeff}
    W_out = [
        {0: 1, 3: 1},     # M1: C00+C11
        {2: 1, 3: -1},    # M2: C10-C11
        {1: 1, 3: 1},     # M3: C01+C11
        {0: 1, 2: 1},     # M4: C00+C10
        {0: -1, 1: 1},    # M5: -C00+C01
        {3: 1},           # M6: C11
        {0: 1}            # M7: C00
    ]
    
    # Helper to expand a block combination and inner vector into a full vector
    def expand(block_combos, inner_vec, size):
        # size is 16 for 4x4
        vec = [0] * size
        for block_idx, coeff in block_combos:
            # Block offset
            # 4x4 matrix blocks:
            # 0: 0-3, 1: 4-7, 2: 8-11, 3: 12-15
            offset = block_idx * 4
            for k in range(4):
                vec[offset + k] += coeff * inner_vec[k]
        return vec
    
    col = 0
    # Block 1 (k=0..3)
    for i in range(7):
        for j in range(7):
            u_vec = expand(U_out[i], U_in[j], 16)
            v_vec = expand(V_out[i], V_in[j], 16)
            w_vec = [0] * 16
            for c_block, coeff in W_out[i].items():
                offset = c_block * 4
                for k in range(4):
                    w_vec[offset + k] += coeff * W_in[j][k]
            
            # Fill U, V, W for Block 1
            for r in range(16):
                U[r][col] = u_vec[r]
                V[r][col] = v_vec[r]
                W[r][col] = w_vec[r]
            col += 1
            
    # Block 2 (k=4..7)
    # V and W indices are shifted by 16
    for i in range(7):
        for j in range(7):
            u_vec = expand(U_out[i], U_in[j], 16)
            v_vec = expand(V_out[i], V_in[j], 16)
            w_vec = [0] * 16
            for c_block, coeff in W_out[i].items():
                offset = c_block * 4
                for k in range(4):
                    w_vec[offset + k] += coeff * W_in[j][k]
            
            # Fill U, V, W for Block 2
            for r in range(16):
                U[r][col] = u_vec[r]
                V[r + 16][col] = v_vec[r]
                W[r + 16][col] = w_vec[r]
            col += 1

    return {"U": U, "V": V, "W": W}
