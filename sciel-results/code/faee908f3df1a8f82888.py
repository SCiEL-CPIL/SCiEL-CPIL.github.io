def construct():
    n, m, p = 4, 4, 9
    
    # Part 1: 4x4x8 using two 4x4x4 Strassen
    # Rank 98
    U1, V1, W1 = get_strassen_4x4x4()
    
    # We need to map the 4x4x4 result to the first 8 columns of the 4x9 result.
    # B is 4x9. We split B into B1 (4x8) and B2 (4x1).
    # B1 is two 4x4 blocks? No, B1 is 4x8.
    # We can view B1 as two 4x4 matrices side-by-side?
    # A * B1 = A * [B1a, B1b] = [A*B1a, A*B1b].
    # So we use two 4x4x4 decompositions.
    
    U_total = np.zeros((16, 114))
    V_total = np.zeros((36, 114))
    W_total = np.zeros((36, 114))
    
    # First 4x4x4 for columns 0-3 of B
    # U1 is 16x49
    # V1 is 16x49 (maps B 4x4 to products)
    # W1 is 16x49 (maps products to C 4x4)
    
    # Copy U1 to U_total[:, 0:49]
    U_total[:, 0:49] = U1
    
    # V1 corresponds to B1a (cols 0-3 of B).
    # Global V is 36x114. B is 4x9.
    # V1 row j (0..3) maps to global row j*9 + 0..3?
    # V1 is 16x49. Rows are flattened 4x4 B.
    # Row index in V1: r*4+c.
    # Global row index: r*9 + c.
    for r in range(4):
        for c in range(4):
            V_total[r*9 + c, 0:49] = V1[r*4 + c, 0:49]
            
    # W1 corresponds to C1a (cols 0-3 of C).
    # W1 is 16x49. Rows are flattened 4x4 C.
    # Row index in W1: r*4+c.
    # Global row index: (c_out)*4 + r?
    # Task says W[k*n+i, r]. k is col, i is row.
    # So global row index: k*4 + i.
    # In W1, row index is k*4 + i (for 4x4 C).
    # So W_total[k*4 + i, 0:49] = W1[k*4 + i, 0:49] for k=0..3.
    for k in range(4):
        for i in range(4):
            W_total[k*4 + i, 0:49] = W1[k*4 + i, 0:49]
            
    # Second 4x4x4 for columns 4-7 of B
    # Same U1
    U_total[:, 49:98] = U1
    
    # V1 maps to B1b (cols 4-7 of B).
    for r in range(4):
        for c in range(4):
            V_total[r*9 + (c+4), 49:98] = V1[r*4 + c, 49:98]
            
    # W1 maps to C1b (cols 4-7 of C).
    for k in range(4):
        for i in range(4):
            W_total[(k+4)*4 + i, 49:98] = W1[k*4 + i, 49:98]
            
    # Part 2: 4x4x1 for column 8 of B
    # Rank 16
    for k in range(16):
        i = k // 4
        j = k % 4
        col_idx = 98 + k
        
        U_total[i*4 + j, col_idx] = 1
        V_total[j*9 + 8, col_idx] = 1
        W_total[8*4 + i, col_idx] = 1
        
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
