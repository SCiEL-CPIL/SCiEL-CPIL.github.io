def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Standard 2x2x2 Strassen: returns list of (u_vec, v_vec, w_vec) for rank-1 terms
    # u_vec, v_vec, w_vec are length-4 vectors (flattened 2x2)
    # Flattening: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # S1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # S2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # S3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # S4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # S5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # S6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # S7
    ]
    
    # Block-level Strassen terms for 4x4x4:
    # Each term is (P_block_indices, Q_block_indices, sign, C_block)
    # where P and Q are 2x2 blocks of A and B respectively
    # The product P*Q contributes to C_block with the given sign
    
    # M1 = (A00 + A11)(B00 + B11) -> contributes to C00 (+), C11 (-)
    # M2 = (A10 + A11)B00 -> contributes to C10 (+), C11 (-)
    # M3 = A00(B11 - B10) -> contributes to C01 (+), C11 (-)
    # M4 = A11(B01 - B00) -> contributes to C00 (-), C10 (+)
    # M5 = (A00 + A01)B11 -> contributes to C00 (-), C01 (+)
    # M6 = (A10 - A00)(B00 + B01) -> contributes to C11 (+)
    # M7 = (A01 - A11)(B10 + B11) -> contributes to C00 (+)
    
    # For each block-level term, I need to specify:
    # - The 2x2 sub-blocks of A that form P (with signs)
    # - The 2x2 sub-blocks of B that form Q (with signs)
    # - Which C blocks this contributes to, and with what sign
    
    # Let me define each term as a list of (a_blocks, b_blocks, c_blocks, sign)
    # where a_blocks is a list of (block_idx, coeff) for A
    # b_blocks is a list of (block_idx, coeff) for B
    # c_blocks is a list of (block_idx, coeff) for C
    
    # Block indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    block_strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        {
            'a': [(0, 1), (3, 1)],   # A00 + A11
            'b': [(0, 1), (3, 1)],   # B00 + B11
            'c': [(0, 1), (3, -1)]   # C00 +, C11 -
        },
        # M2 = (A10 + A11)B00
        {
            'a': [(2, 1), (3, 1)],   # A10 + A11
            'b': [(0, 1)],           # B00
            'c': [(2, 1), (3, -1)]   # C10 +, C11 -
        },
        # M3 = A00(B11 - B10)
        {
            'a': [(0, 1)],           # A00
            'b': [(3, 1), (2, -1)],  # B11 - B10
            'c': [(1, 1), (3, -1)]   # C01 +, C11 -
        },
        # M4 = A11(B01 - B00)
        {
            'a': [(3, 1)],           # A11
            'b': [(1, 1), (0, -1)],  # B01 - B00
            'c': [(0, -1), (2, 1)]   # C00 -, C10 +
        },
        # M5 = (A00 + A01)B11
        {
            'a': [(0, 1), (1, 1)],   # A00 + A01
            'b': [(3, 1)],           # B11
            'c': [(0, -1), (1, 1)]   # C00 -, C01 +
        },
        # M6 = (A10 - A00)(B00 + B01)
        {
            'a': [(2, 1), (0, -1)],  # A10 - A00
            'b': [(0, 1), (1, 1)],   # B00 + B01
            'c': [(3, 1)]            # C11 +
        },
        # M7 = (A01 - A11)(B10 + B11)
        {
            'a': [(1, 1), (3, -1)],  # A01 - A11
            'b': [(2, 1), (3, 1)],   # B10 + B11
            'c': [(0, 1)]            # C00 +
        }
    ]
    
    # Get flattened indices for a 2x2 block in a 4x4 matrix
    def get_block_indices(block_idx):
        # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        bi = block_idx // 2
        bj = block_idx % 2
        start_r = bi * 2
        start_c = bj * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = start_r + dr
                c = start_c + dc
                indices.append(r * 4 + c)
        return indices
    
    # Precompute
    block_indices = [get_block_indices(i) for i in range(4)]
    
    # Rank for 4x4x8: 2 blocks * 7 block-strassen * 7 element-strassen = 98
    rank = 98
    
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 98
    
    for block in range(2):
        base_col = block * 49
        for bterm_idx, bterm in enumerate(block_strassen_terms):
            for s_idx, (u_2, v_2, w_2) in enumerate(strassen_2x2):
                col = base_col + bterm_idx * 7 + s_idx
                
                # U: accumulate contributions from A blocks
                for a_block_idx, a_coeff in bterm['a']:
                    a_indices = block_indices[a_block_idx]
                    for k, idx in enumerate(a_indices):
                        U[idx][col] += a_coeff * u_2[k]
                
                # V: accumulate contributions from B blocks
                for b_block_idx, b_coeff in bterm['b']:
                    b_indices = block_indices[b_block_idx]
                    for k, idx in enumerate(b_indices):
                        b_row = idx // 4
                        b_col_local = idx % 4
                        b_col_global = block * 4 + b_col_local
                        v_flat_idx = b_row * p + b_col_global
                        V[v_flat_idx][col] += b_coeff * v_2[k]
                
                # W: accumulate contributions from C blocks
                for c_block_idx, c_coeff in bterm['c']:
                    c_indices = block_indices[c_block_idx]
                    for k, idx in enumerate(c_indices):
                        c_row = idx // 4
                        c_col_local = idx % 4
                        c_col_global = block * 4 + c_col_local
                        w_flat_idx = c_col_global * n + c_row
                        W[w_flat_idx][col] += c_coeff * w_2[k]
    
    return {"U": U, "V": V, "W": W}
