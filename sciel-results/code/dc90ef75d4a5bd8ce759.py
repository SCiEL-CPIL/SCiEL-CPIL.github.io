DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # U: 4 elements (A00, A01, A10, A11)
    # V: 4 elements per block (B00, B01, B10, B11)
    # W: 4 elements per block (C00, C01, C10, C11)
    
    # 7 Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {idx: val}
    # v_coeffs: dict {idx: val}
    # w_coeffs: dict {idx: val}
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})        # M7
    ]
    
    col = 0
    for b in range(num_blocks):
        # Global index mappings for V and W
        # V: B00, B01, B10, B11
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        # W: C00, C01, C10, C11
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
