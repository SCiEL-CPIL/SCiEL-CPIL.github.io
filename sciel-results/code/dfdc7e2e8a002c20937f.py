import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] = coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    # Strassen 2x2x2 terms
    strassen_2x2_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # 2x2x3 terms (Rank 11: 7 Strassen + 4 Schoolbook for 3rd col)
    # Schoolbook for 3rd col: x00*y02, x01*y12, x10*y02, x11*y12
    schoolbook_2x2x1_terms = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1})
    ]
    
    # Combine for 2x2x3
    # Note: The 3rd column of Y is at index 2 in the 2x3 block.
    # In the 2x2x2 Strassen, we used cols 0 and 1.
    # Here we need to map the 3rd column.
    
    # We construct the 4x4x6 decomposition using Kronecker product logic.
    # Block indices: I, J, K in {0, 1}
    # Internal indices: i, j, k
    # A[I, J] is 2x2. B[J, K] is 2x3. C[I, K] is 2x3.
    
    # We iterate over the 7 Strassen terms for the block structure.
    # For each block term, we iterate over the 11 terms for the internal structure.
    
    for bi, bj, bk_u, bk_v, bk_w in [
        ((0,0), (0,0), (0,0)), # Placeholder, we use the term dicts
    ]:
        pass
        
    # Explicit Kronecker construction
    for u_block, v_block, w_block in strassen_2x2_terms:
        for u_int, v_int, w_int in strassen_2x2_terms + schoolbook_2x2x1_terms:
            # Combine indices
            u_full = {}
            v_full = {}
            w_full = {}
            
            # u_block maps (bi, bj) to coeff
            # u_int maps (i, j) to coeff
            # Full index (I, J) where I = 2*bi + i, J = 2*bj + j
            for (bi, bj), cb in u_block.items():
                for (i, j), ci in u_int.items():
                    I = 2 * bi + i
                    J = 2 * bj + j
                    u_full[(I, J)] = u_full.get((I, J), 0) + cb * ci
            
            # v_block maps (bj, bk) to coeff
            # v_int maps (j, k) to coeff
            # Full index (J, K) where J = 2*bj + j, K = 2*bk + k
            # Note: v_int for schoolbook uses k=0 for the 3rd col?
            # In 2x2x1, the output is 2x1. The index k is 0.
            # In 2x2x3, the output is 2x3. The index k is 0, 1, 2.
            # The Strassen terms use k=0, 1.
            # The Schoolbook terms for the 3rd col should use k=2.
            
            # We need to distinguish between the 7 Strassen terms (using k=0,1) 
            # and the 4 Schoolbook terms (using k=2).
            
            # Let's redefine the 11 terms with explicit k indices.
            pass
            
    # Redefine terms with explicit k indices for the 2x2x3 case
    # Strassen terms (k in 0, 1)
    strassen_2x2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Schoolbook for 3rd col (k=2)
    schoolbook_2x2x1_col2 = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}), # x00*y02 -> Z02
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}), # x01*y12 -> Z02
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}), # x10*y02 -> Z12
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1})  # x11*y12 -> Z12
    ]
    
    # In the schoolbook terms above, the 'w' index 0 corresponds to the 3rd column of the 2x3 block.
    # So we must map w_index 0 to global k = 2*bk + 2? No.
    # The block K is 0 or 1.
    # If K=0, the block is cols 0,1,2 of B? No.
    # B is 4x6.
    # Block rows: 0 (rows 0-1), 1 (rows 2-3).
    # Block cols: 0 (cols 0-2), 1 (cols 3-5).
    # So if bk=0, internal k=0,1,2 map to global k=0,1,2.
    # If bk=1, internal k=0,1,2 map to global k=3,4,5.
    
    for u_block, v_block, w_block in strassen_2x2_terms:
        for u_int, v_int, w_int in strassen_2x2x2 + schoolbook_2x2x1_col2:
            u_full = {}
            v_full = {}
            w_full = {}
            
            for (bi, bj), cb in u_block.items():
                for (i, j), ci in u_int.items():
                    I = 2 * bi + i
                    J = 2 * bj + j
                    u_full[(I, J)] = u_full.get((I, J), 0) + cb * ci
            
            for (bj, bk), cb in v_block.items():
                for (j, k), ci in v_int.items():
                    J = 2 * bj + j
                    K = 2 * bk + k
                    v_full[(J, K)] = v_full.get((J, K), 0) + cb * ci
            
            for (bi, bk), cb in w_block.items():
                for (i, k), ci in w_int.items():
                    I = 2 * bi + i
                    K = 2 * bk + k
                    w_full[(I, K)] = w_full.get((I, K), 0) + cb * ci
            
            add_term(u_full, v_full, w_full)
            
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
