"""Matrix multiplication 3,4,6 rank-minimal decomposition."""

def construct():
    n, m, p = 3, 4, 6
    
    # We will build the decomposition step by step.
    # U: (12, rank), V: (24, rank), W: (18, rank)
    # We accumulate vectors in lists.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # This is not quite right for Strassen which has multiple non-zeros.
        # Better to pass full local vectors and mappings.
        pass

    def add_strassen(a_map, b_map, c_map):
        # a_map: dict local_idx -> global_idx for A (2x2)
        # b_map: dict local_idx -> global_idx for B (2x2)
        # c_map: dict local_idx -> global_idx for C (2x2)
        # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Strassen terms
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b11-b01)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        terms = [
            # (u_vec, v_vec, w_vec)
            ([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2
            ([1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0]),  # M3
            ([0, 0, 0, 1], [1, 0, 1, 0], [0, 0, 1, 0]),  # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, -1]), # M5
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])  # M7
        ]
        
        for u_l, v_l, w_l in terms:
            u_g = [0] * (n*m)
            v_g = [0] * (m*p)
            w_g = [0] * (p*n)
            
            for li, val in enumerate(u_l):
                if val != 0:
                    u_g[a_map[li]] = val
            for li, val in enumerate(v_l):
                if val != 0:
                    v_g[b_map[li]] = val
            for li, val in enumerate(w_l):
                if val != 0:
                    w_g[c_map[li]] = val
                    
            U_cols.append(u_g)
            V_cols.append(v_g)
            W_cols.append(w_g)

    def add_schoolbook_2x2x2(a_map, b_map, c_map):
        # Just 4 terms
        for i in range(2):
            for j in range(2):
                u_g = [0] * (n*m)
                v_g = [0] * (m*p)
                w_g = [0] * (p*n)
                u_g[a_map[i*2+j]] = 1
                v_g[b_map[i*2+j]] = 1
                w_g[c_map[i*2+j]] = 1
                U_cols.append(u_g)
                V_cols.append(v_g)
                W_cols.append(w_g)

    def add_schoolbook_2x2x1(a_map, b_map, c_map):
        # b_map has 2 entries (row 0 col 0, row 1 col 0)
        # c_map has 2 entries (row 0 col 0, row 1 col 0)
        # a_map has 4 entries
        for i in range(2):
            for k in range(2): # k is the row of B, which is the col of A? No.
                # A is 2x2, B is 2x1, C is 2x1.
                # C_i = sum_j A_ij B_j
                # Term A_i0 B_0 -> C_i
                # Term A_i1 B_1 -> C_i
                pass
        # Let's just do it explicitly
        # A indices: 0,1,2,3 (0,0; 0,1; 1,0; 1,1)
        # B indices: 0,1 (0,0; 1,0)
        # C indices: 0,1 (0,0; 1,0)
        
        terms = [
            (0, 0, 0), # A00 * B0 -> C0
            (1, 1, 0), # A01 * B1 -> C0
            (2, 0, 1), # A10 * B0 -> C1
            (3, 1, 1)  # A11 * B1 -> C1
        ]
        for ai, bi, ci in terms:
            u_g = [0] * (n*m)
            v_g = [0] * (m*p)
            w_g = [0] * (p*n)
            u_g[a_map[ai]] = 1
            v_g[b_map[bi]] = 1
            w_g[c_map[ci]] = 1
            U_cols.append(u_g)
            V_cols.append(v_g)
            W_cols.append(w_g)

    def add_schoolbook_1x4x6(a_row_idx, b_map_4x6, c_map_1x6):
        # a_row_idx is the global row index in A
        # b_map_4x6: list of 24 global indices for B
        # c_map_1x6: list of 6 global indices for C
        for j in range(4):
            for k in range(6):
                u_g = [0] * (n*m)
                v_g = [0] * (m*p)
                w_g = [0] * (p*n)
                u_g[a_row_idx * m + j] = 1
                v_g[b_map_4x6[j*6+k]] = 1
                w_g[c_map_1x6[k]] = 1
                U_cols.append(u_g)
                V_cols.append(v_g)
                W_cols.append(w_g)

    # 1. C_0: Row 0 of C
    # A row 0: indices 0,1,2,3
    # B full: indices 0..23
    # C row 0: indices k*3 + 0 for k in 0..5 -> 0, 3, 6, 9, 12, 15
    b_map_full = [j*6+k for j in range(4) for k in range(6)]
    c_map_row0 = [k*3+0 for k in range(6)]
    add_schoolbook_1x4x6(0, b_map_full, c_map_row0)

    # 2. C_1,2: Rows 1,2 of C
    # A rows 1,2: indices 4..11
    # Split A into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B into B1 (rows 0,1) and B2 (rows 2,3)
    
    # A1 map: 4,5,8,9
    # A2 map: 6,7,10,11
    # B1 map: 0..5, 6..11
    # B2 map: 12..17, 18..23
    # C map for rows 1,2:
    # C_1k: k*3+1
    # C_2k: k*3+2
    
    # Sub-problem 2a: A1 * B1 (2x2x6)
    # Split B1 into B1a (cols 0,1,2) and B1b (cols 3,4,5)
    # B1a map: 0,1,2, 6,7,8
    # B1b map: 3,4,5, 9,10,11
    
    # 2a-i: A1 * B1a (2x2x3)
    # Split B1a into B1a0 (cols 0,1) and B1a1 (col 2)
    # B1a0 map: 0,1, 6,7
    # B1a1 map: 2, 8
    
    # 2a-i-A: A1 * B1a0 (2x2x2) -> Strassen
    a_map_1 = [4, 5, 8, 9]
    b_map_1a0 = [0, 1, 6, 7]
    c_map_12_01 = [1, 4, 2, 5] # C10, C11, C20, C21
    add_strassen(a_map_1, b_map_1a0, c_map_12_01)
    
    # 2a-i-B: A1 * B1a1 (2x2x1) -> Schoolbook
    b_map_1a1 = [2, 8]
    c_map_12_2 = [7, 8] # C12, C22
    add_schoolbook_2x2x1(a_map_1, b_map_1a1, c_map_12_2)
    
    # 2a-ii: A1 * B1b (2x2x3)
    # Split B1b into B1b0 (cols 3,4) and B1b1 (col 5)
    # B1b0 map: 3,4, 9,10
    # B1b1 map: 5, 11
    
    # 2a-ii-A: A1 * B1b0 (2x2x2) -> Strassen
    b_map_1b0 = [3, 4, 9, 10]
    c_map_12_34 = [10, 13, 11, 14] # C13, C14, C23, C24
    add_strassen(a_map_1, b_map_1b0, c_map_12_34)
    
    # 2a-ii-B: A1 * B1b1 (2x2x1) -> Schoolbook
    b_map_1b1 = [5, 11]
    c_map_12_5 = [16, 17] # C15, C25
    add_schoolbook_2x2x1(a_map_1, b_map_1b1, c_map_12_5)
    
    # Sub-problem 2b: A2 * B2 (2x2x6)
    # A2 map: 6,7,10,11
    # B2 map: 12..17, 18..23
    # Split B2 into B2a (cols 0,1,2) and B2b (cols 3,4,5)
    # B2a map: 12,13,14, 18,19,20
    # B2b map: 15,16,17, 21,22,23
    
    # 2b-i: A2 * B2a (2x2x3)
    # Split B2a into B2a0 (cols 0,1) and B2a1 (col 2)
    # B2a0 map: 12,13, 18,19
    # B2a1 map: 14, 20
    
    # 2b-i-A: A2 * B2a0 (2x2x2) -> Strassen
    a_map_2 = [6, 7, 10, 11]
    b_map_2a0 = [12, 13, 18, 19]
    c_map_12_01 = [1, 4, 2, 5]
    add_strassen(a_map_2, b_map_2a0, c_map_12_01)
    
    # 2b-i-B: A2 * B2a1 (2x2x1) -> Schoolbook
    b_map_2a1 = [14, 20]
    c_map_12_2 = [7, 8]
    add_schoolbook_2x2x1(a_map_2, b_map_2a1, c_map_12_2)
    
    # 2b-ii: A2 * B2b (2x2x3)
    # Split B2b into B2b0 (cols 3,4) and B2b1 (col 5)
    # B2b0 map: 15,16, 21,22
    # B2b1 map: 17, 23
    
    # 2b-ii-A: A2 * B2b0 (2x2x2) -> Strassen
    b_map_2b0 = [15, 16, 21, 22]
    c_map_12_34 = [10, 13, 11, 14]
    add_strassen(a_map_2, b_map_2b0, c_map_12_34)
    
    # 2b-ii-B: A2 * B2b1 (2x2x1) -> Schoolbook
    b_map_2b1 = [17, 23]
    c_map_12_5 = [16, 17]
    add_schoolbook_2x2x1(a_map_2, b_map_2b1, c_map_12_5)
    
    # Convert lists to matrices
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U[i][r] = U_cols[r][i]
        for i in range(m*p):
            V[i][r] = V_cols[r][i]
        for i in range(p*n):
            W[i][r] = W_cols[r][i]
            
    return {"U": U, "V": V, "W": W}
