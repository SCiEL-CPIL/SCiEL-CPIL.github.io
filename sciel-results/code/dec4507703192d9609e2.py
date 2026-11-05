def construct():
    n, m, p = 4, 4, 9
    
    # Generate 4x4x4 Strassen
    U1, V1, W1 = strassen_4x4x4()
    
    # Generate 4x4x1 Schoolbook
    rank1 = 16
    U3 = [[0]*rank1 for _ in range(16)]
    V3 = [[0]*rank1 for _ in range(4)] # 4 rows in B for 1 col
    W3 = [[0]*rank1 for _ in range(4)] # 4 rows in C for 1 col
    
    col = 0
    for i in range(4):
        for j in range(4):
            U3[i*4+j][col] = 1
            V3[j][col] = 1
            W3[i][col] = 1
            col += 1
            
    # Assemble
    total_rank = 49 + 49 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Part 1: 4x4x4 (cols 0-3 of B)
    for r in range(16):
        for c in range(49):
            U[r][c] = U1[r][c]
            V[r][c] = V1[r][c] # B[0:4, 0:4] -> rows 0-15
            W[r][c] = W1[r][c] # C[0:4, 0:4] -> rows 0-15
    
    # Part 2: 4x4x4 (cols 4-7 of B)
    for r in range(16):
        for c in range(49):
            U[r][49+c] = U1[r][c]
            V[r+16][49+c] = V1[r][c] # B[0:4, 4:8] -> rows 16-31
            W[r+16][49+c] = W1[r][c] # C[0:4, 4:8] -> rows 16-31
            
    # Part 3: 4x4x1 (col 8 of B)
    for r in range(16):
        for c in range(16):
            U[r][98+c] = U3[r][c]
            # V3 has 4 rows. B[0:4, 8:9] -> rows 32-35
            # V3[r] corresponds to B[r, 0] in local 4x1.
            # Global B index is r*9 + 8.
            # But V matrix is indexed by global B index.
            # V[global_b_idx][col]
            # global_b_idx = r*9 + 8.
            # So V[r*9+8][98+c] = V3[r][c]
            V[r*9+8][98+c] = V3[r][c]
            
            # W3 has 4 rows. C[0:4, 8:9] -> rows 32-35.
            # W3[r] corresponds to C[r, 0] in local 4x1.
            # Global C index is 8*4 + r = 32 + r.
            W[32+r][98+c] = W3[r][c]
            
    return {"U": U, "V": V, "W": W}
