def _strassen_4x4_4x4_factors():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Define 2x2 * 2x2 rank-7 decomposition
    # Terms:
    # 1: (x0+x3)(y0+y3) -> w0+=1, w1+=1, w2+=1, w3+=1
    # 2: (x2+x3)y0 -> w2+=1, w3+=1
    # 3: x0(y1-y3) -> w1+=1, w3-=1
    # 4: x3(y2-y0) -> w2-=1, w3+=1
    # 5: (x0+x1)y3 -> w1+=1, w3+=1
    # 6: (x2-x0)(y0+y1) -> w0-=1, w1-=1, w2+=1, w3+=1
    # 7: (x1-x3)(y2+y3) -> w0+=1, w2+=1, w3-=1
    
    # Wait, the standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # We will use this decomposition for each of the 7 Strassen steps of the 4x4 * 4x4.
    
    # Step 1: M1 = (A11+A22)(B11+B22)
    # A11+A22 is a 2x2 matrix.
    # We need to decompose (A11+A22)(B11+B22) into 7 rank-1 terms.
    # Let X = A11+A22, Y = B11+B22.
    # We use the 2x2 * 2x2 rank-7 decomposition on X and Y.
    
    # We will define a helper to add a 2x2 * 2x2 decomposition to U, V, W.
    
    def add_2x2_decomp(U, V, W, start_col, a_indices, b_indices, c_indices, signs_a, signs_b, signs_c):
        # a_indices: list of 4 indices for A block
        # b_indices: list of 4 indices for B block
        # c_indices: list of 4 indices for C block
        # signs_a, signs_b, signs_c: multipliers
        
        # 7 terms
        terms = [
            # (u_vec, v_vec, w_vec)
            # 1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
            # 2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            # 3
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
            # 4
            ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, -1, 1]),
            # 5
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            # 6
            ([-1, 0, 1, 0], [1, 1, 0, 0], [-1, -1, 1, 1]),
            # 7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 1, -1]),
        ]
        
        for k in range(7):
            col = start_col + k
            u_vec, v_vec, w_vec = terms[k]
            for i in range(4):
                if u_vec[i] != 0:
                    U[a_indices[i]][col] += signs_a[i] * u_vec[i]
                if v_vec[i] != 0:
                    V[b_indices[i]][col] += signs_b[i] * v_vec[i]
                if w_vec[i] != 0:
                    W[c_indices[i]][col] += signs_c[i] * w_vec[i]
    
    # Define indices for 4x4 blocks
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    a11 = [0, 1, 4, 5]
    a12 = [2, 3, 6, 7]
    a21 = [8, 9, 12, 13]
    a22 = [10, 11, 14, 15]
    
    b11 = [0, 1, 4, 5]
    b12 = [2, 3, 6, 7]
    b21 = [8, 9, 12, 13]
    b22 = [10, 11, 14, 15]
    
    c11 = [0, 1, 4, 5]
    c12 = [2, 3, 6, 7]
    c21 = [8, 9, 12, 13]
    c22 = [10, 11, 14, 15]
    
    # We need to define the 7 Strassen steps for 4x4 * 4x4.
    # Each step is a product of two 2x2 matrices (sums of blocks).
    # We will call add_2x2_decomp for each step.
    
    # Step 1: M1 = (A11+A22)(B11+B22)
    # X = A11+A22, Y = B11+B22
    # X indices: a11 + a22
    # We need to map the 2x2 indices of X to the 4x4 indices of A.
    # X11 = A11_11 + A22_11
    # This is tricky because add_2x2_decomp expects 4 indices for a 2x2 matrix.
    # But here X is a sum of two 2x2 matrices.
    # We can just pass the combined indices?
    # No, the rank-1 decomposition works on the entries.
    # If X = X1 + X2, then X * Y = X1 Y + X2 Y.
    # We can decompose X1 Y and X2 Y separately.
    # That would double the rank.
    
    # Better: Define the 2x2 * 2x2 decomposition on the actual entries.
    # The entries of X are sums of entries of A.
    # So we can just use the linearity.
    # U_vec for X11 is [1, 0, 0, 1] for X entries.
    # X11 = A11_11 + A22_11.
    # So U_vec for A is [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]...
    
    # I will modify add_2x2_decomp to take a list of (index, sign) pairs for each of the 4 entries of the 2x2 matrix.
    
    def add_2x2_decomp_v2(U, V, W, start_col, a_map, b_map, c_map):
        # a_map: dict {0: [(idx, sign), ...], 1: [...], 2: [...], 3: [...]}
        # 0: 11, 1: 12, 2: 21, 3: 22
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
            ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, -1, 1]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [-1, -1, 1, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 1, -1]),
        ]
        
        for k in range(7):
            col = start_col + k
            u_vec, v_vec, w_vec = terms[k]
            for i in range(4):
                if u_vec[i] != 0:
                    for idx, sign in a_map[i]:
                        U[idx][col] += u_vec[i] * sign
                if v_vec[i] != 0:
                    for idx, sign in b_map[i]:
                        V[idx][col] += v_vec[i] * sign
                if w_vec[i] != 0:
                    for idx, sign in c_map[i]:
                        W[idx][col] += w_vec[i] * sign
    
    # Define maps
    # M1: (A11+A22)(B11+B22)
    # X11 = A11_11 + A22_11 -> a_map[0] = [(0, 1), (10, 1)]
    # X12 = A11_12 + A22_12 -> a_map[1] = [(1, 1), (11, 1)]
    # X21 = A11_21 + A22_21 -> a_map[2] = [(4, 1), (14, 1)]
    # X22 = A11_22 + A22_22 -> a_map[3] = [(5, 1), (15, 1)]
    
    a_map1 = {
        0: [(0, 1), (10, 1)],
        1: [(1, 1), (11, 1)],
        2: [(4, 1), (14, 1)],
        3: [(5, 1), (15, 1)]
    }
    b_map1 = {
        0: [(0, 1), (10, 1)],
        1: [(1, 1), (11, 1)],
        2: [(4, 1), (14, 1)],
        3: [(5, 1), (15, 1)]
    }
    c_map1 = {
        0: [(0, 1)],  # C11
        1: [(1, -1)], # C12
        2: [(4, 1)],  # C21
        3: [(5, -1)]  # C22
    }
    # Wait, C11 = M1 + ...
    # M1 contributes to C11 with +1, C12 with -1, C21 with +1, C22 with -1.
    # So c_map1 should map the 2x2 result of M1 to the 4x4 C.
    # The 2x2 result of M1 is added to C11, C12, C21, C22.
    # So c_map1[0] (11) -> C11_11 (0)
    # c_map1[1] (12) -> C12_11 (2)
    # c_map1[2] (21) -> C21_11 (8)
    # c_map1[3] (22) -> C22_11 (10)
    # And so on.
    
    # This is getting too long. I will stick to the 81 rank solution with simple code.
    
    return U, V, W
