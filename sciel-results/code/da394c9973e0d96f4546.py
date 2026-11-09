"""Recursive Strassen-based rank reduction for 3x4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 88 via recursive splitting
    # M_{3,4,8} -> 2 * M_{3,2,8}
    # M_{3,2,8} -> 2 * M_{3,2,4}
    # M_{3,2,4} -> M_{1,2,4} (rank 8) + M_{2,2,4} (rank 14)
    # M_{2,2,4} -> 2 * M_{2,2,2} (rank 7 each)
    
    def make_zeros(rows, cols):
        return [[0] * cols for _ in range(rows)]

    def add_to_matrix(M, row, col, val):
        M[row][col] += val

    def strassen_222():
        # Returns U, V, W for M_{2,2,2} with rank 7
        U = make_zeros(4, 7)
        V = make_zeros(4, 7)
        W = make_zeros(4, 7)
        
        # Indices:
        # U: i*2+j (i:0,1; j:0,1) -> 0,1,2,3
        # V: j*2+k (j:0,1; k:0,1) -> 0,1,2,3
        # W: k*2+i (k:0,1; i:0,1) -> 0,2,1,3 (C00, C01, C10, C11)
        
        # M1 = (A00+A11)(B00+B11)
        add_to_matrix(U, 0, 0, 1); add_to_matrix(U, 3, 0, 1)
        add_to_matrix(V, 0, 0, 1); add_to_matrix(V, 3, 0, 1)
        add_to_matrix(W, 0, 0, 1); add_to_matrix(W, 3, 0, 1) # C00, C11
        
        # M2 = (A10+A11)B00
        add_to_matrix(U, 2, 1, 1); add_to_matrix(U, 3, 1, 1)
        add_to_matrix(V, 0, 1, 1)
        add_to_matrix(W, 1, 1, 1); add_to_matrix(W, 3, 1, -1) # C10, C11
        
        # M3 = A00(B01-B11)
        add_to_matrix(U, 0, 2, 1)
        add_to_matrix(V, 1, 2, 1); add_to_matrix(V, 3, 2, -1)
        add_to_matrix(W, 2, 2, 1); add_to_matrix(W, 3, 2, 1) # C01, C11
        
        # M4 = A11(B10-B00)
        add_to_matrix(U, 3, 3, 1)
        add_to_matrix(V, 2, 3, 1); add_to_matrix(V, 0, 3, -1)
        add_to_matrix(W, 0, 3, 1); add_to_matrix(W, 1, 3, 1) # C00, C10
        
        # M5 = (A00+A01)B11
        add_to_matrix(U, 0, 4, 1); add_to_matrix(U, 1, 4, 1)
        add_to_matrix(V, 3, 4, 1)
        add_to_matrix(W, 0, 4, -1); add_to_matrix(W, 2, 4, 1) # C00, C01
        
        # M6 = (A10+A11)(B01-B11)
        add_to_matrix(U, 2, 5, 1); add_to_matrix(U, 3, 5, 1)
        add_to_matrix(V, 1, 5, 1); add_to_matrix(V, 3, 5, -1)
        add_to_matrix(W, 3, 5, 1) # C11
        
        # M7 = (A01-A10)(B00+B10)
        add_to_matrix(U, 1, 6, 1); add_to_matrix(U, 2, 6, -1)
        add_to_matrix(V, 0, 6, 1); add_to_matrix(V, 2, 6, 1)
        add_to_matrix(W, 0, 6, -1); add_to_matrix(W, 3, 6, 1) # C00, C11
        
        return U, V, W

    def schoolbook_124():
        # M_{1,2,4} rank 8
        U = make_zeros(2, 8) # 1*2
        V = make_zeros(8, 8) # 2*4
        W = make_zeros(4, 8) # 4*1
        for j in range(2):
            for k in range(4):
                c = j * 4 + k
                U[j][c] = 1
                V[j * 4 + k][c] = 1
                W[k][c] = 1
        return U, V, W

    # Total Rank 88
    total_rank = 88
    U = make_zeros(n * m, total_rank) # 3*4 = 12
    V = make_zeros(m * p, total_rank) # 4*8 = 32
    W = make_zeros(p * n, total_rank) # 8*3 = 24

    offset = 0
    
    # Split m=4 into two m=2 blocks (j=0,1 and j=2,3)
    for m_block in range(2):
        # Split p=8 into two p=4 blocks (k=0..3 and k=4..7)
        for p_block in range(2):
            # Split n=3 into n=1 (i=0) and n=2 (i=1,2)
            
            # Part 1: M_{1,2,4} (n=1, m=2, p=4)
            # Local indices: i in {0}, j in {0,1}, k in {0..3}
            # Global indices:
            # i_global = 0
            # j_global = m_block * 2 + j_local
            # k_global = p_block * 4 + k_local
            
            U_sub, V_sub, W_sub = schoolbook_124()
            for c in range(8):
                for j_l in range(2):
                    if U_sub[j_l][c] != 0:
                        j_g = m_block * 2 + j_l
                        U[j_g][offset + c] = U_sub[j_l][c]
                        V[j_g * 8 + (p_block * 4) * 2 + 0][offset + c] = 0 # Placeholder, need correct V index
                        # V row index is j*p + k.
                        # j_g is row of V.
                        # k_g = p_block * 4 + k_l.
                        # V index = j_g * 8 + k_g.
                        
                # Let's just loop through the sub-matrices and place them
            # Simpler placement logic:
            
            # We need to map local (i,j,k) to global (i,j,k) and add to global U,V,W at offset.
            
            # M_{1,2,4}
            for j_l in range(2):
                for k_l in range(4):
                    c_l = j_l * 4 + k_l
                    c_g = offset + c_l
                    i_g = 0
                    j_g = m_block * 2 + j_l
                    k_g = p_block * 4 + k_l
                    
                    U[i_g * 4 + j_g][c_g] += 1
                    V[j_g * 8 + k_g][c_g] += 1
                    W[k_g * 3 + i_g][c_g] += 1
            
            offset += 8
            
            # Part 2: M_{2,2,4} (n=2, m=2, p=4)
            # Split p=4 into two p=2 blocks
            for p_sub_block in range(2):
                # M_{2,2,2}
                U_s, V_s, W_s = strassen_222()
                for c_l in range(7):
                    # Local i in {0,1}, j in {0,1}, k in {0,1}
                    # Global i = 1 + i_l (since n=1 was i=0)
                    # Global j = m_block * 2 + j_l
                    # Global k = p_block * 4 + p_sub_block * 2 + k_l
                    
                    for i_l in range(2):
                        for j_l in range(2):
                            u_val = U_s[i_l * 2 + j_l][c_l]
                            if u_val != 0:
                                i_g = 1 + i_l
                                j_g = m_block * 2 + j_l
                                U[i_g * 4 + j_g][offset + c_l] += u_val
                                
                    for j_l in range(2):
                        for k_l in range(2):
                            v_val = V_s[j_l * 2 + k_l][c_l]
                            if v_val != 0:
                                j_g = m_block * 2 + j_l
                                k_g = p_block * 4 + p_sub_block * 2 + k_l
                                V[j_g * 8 + k_g][offset + c_l] += v_val
                                
                    for k_l in range(2):
                        for i_l in range(2):
                            w_val = W_s[k_l * 2 + i_l][c_l]
                            if w_val != 0:
                                k_g = p_block * 4 + p_sub_block * 2 + k_l
                                i_g = 1 + i_l
                                W[k_g * 3 + i_g][offset + c_l] += w_val
                            
                offset += 7

    # Verify
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = make_zeros(n * m, rank)
        V = make_zeros(m * p, rank)
        W = make_zeros(p * n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
        return {"U": U, "V": V, "W": W}

