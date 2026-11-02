def construct():
    n, m, p = 2, 2, 14
    num_blocks = p // m  # 7 blocks of 2x2
    rank = 7 * num_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping A_index (0..3) to coeff
    # v_coeffs: dict mapping B_index (0..3) to coeff
    # w_coeffs: dict mapping C_index (0..3) to coeff
    # Indices:
    # A: 0=00, 1=01, 2=10, 3=11
    # B: 0=00, 1=01, 2=10, 3=11
    # C: 0=00, 1=01, 2=10, 3=11
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})         # M7
    ]
    
    col = 0
    for b in range(num_blocks):
        # Global indices for V and W for this block
        # B is 2x14. Block b covers columns 2*b and 2*b+1.
        # V index = row * p + col
        # B00 -> row 0, col 2*b -> 0 * 14 + 2*b
        # B01 -> row 0, col 2*b+1 -> 0 * 14 + 2*b+1
        # B10 -> row 1, col 2*b -> 1 * 14 + 2*b
        # B11 -> row 1, col 2*b+1 -> 1 * 14 + 2*b+1
        
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # C is 2x14. Output column k corresponds to input column k.
        # W index = col * n + row (trace convention W[k*n+i])
        # C00 -> col 2*b, row 0 -> 2*b * 2 + 0
        # C01 -> col 2*b+1, row 0 -> (2*b+1) * 2 + 0
        # C10 -> col 2*b, row 1 -> 2*b * 2 + 1
        # C11 -> col 2*b+1, row 1 -> (2*b+1) * 2 + 1
        
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                global_v_idx = v_offsets[l_idx]
                V[global_v_idx][col] += val
            for l_idx, val in w_dict.items():
                global_w_idx = w_offsets[l_idx]
                W[global_w_idx][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
