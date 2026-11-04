def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = 2, 4, 5
    
    # Total rank 36
    # 4 blocks of 2x2x2 (Strassen, rank 7)
    # 2 blocks of 2x2x1 (Schoolbook, rank 4)
    total_rank = 4 * 7 + 2 * 4
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen block helper
    # Maps local (i, j, k) to global (i_g, j_g, k_g)
    # i_g = i_off + i, j_g = j_off + j, k_g = k_off + k
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # M1 = (a+d)(e+h)
        # U: (0,0), (1,1) -> 1
        # V: (0,0), (1,1) -> 1
        # W: (0,0), (1,1) -> 1
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # M3 = a(f-h)
        U[i_off * m + j_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = -1
        W[k_off * n + i_off][col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = -1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
    # Schoolbook 2x2x1 block helper
    # C_i = sum_j A_ij B_j
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    def add_schoolbook_2x2x1_block(i_off, j_off, k_off):
        nonlocal col
        # Term 1: A_00 B_0 -> C_0
        U[i_off * m + j_off][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1
        
        # Term 2: A_01 B_1 -> C_0
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1
        
        # Term 3: A_10 B_0 -> C_1
        U[(i_off+1) * m + j_off][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 4: A_11 B_1 -> C_1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4}
    add_schoolbook_2x2x1_block(0, 0, 4)
    
    # Block 4: i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    
    # Block 5: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    
    # Block 6: i in {0,1}, j in {2,3}, k in {4}
    add_schoolbook_2x2x1_block(0, 2, 4)
    
    if time.time() > _deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
