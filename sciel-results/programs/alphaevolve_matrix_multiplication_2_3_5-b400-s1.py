def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block
    # A indices: i in 0..1, j in j_offset..j_offset+1
    # B indices: j in j_offset..j_offset+1, k in k_offset..k_offset+1
    # C indices: i in 0..1, k in k_offset..k_offset+1
    def add_strassen_block(rank_offset, j_offset, k_offset):
        # A vars
        a_idx = 0 * m + j_offset
        b_idx = 0 * m + (j_offset + 1)
        c_idx = 1 * m + j_offset
        d_idx = 1 * m + (j_offset + 1)
        
        # B vars
        e_idx = j_offset * p + k_offset
        f_idx = j_offset * p + (k_offset + 1)
        g_idx = (j_offset + 1) * p + k_offset
        h_idx = (j_offset + 1) * p + (k_offset + 1)
        
        # Ranks
        r0 = rank_offset
        r1 = rank_offset + 1
        r2 = rank_offset + 2
        r3 = rank_offset + 3
        r4 = rank_offset + 4
        r5 = rank_offset + 5
        r6 = rank_offset + 6
        
        # U (L_A)
        U[a_idx][r0] += 1; U[d_idx][r0] += 1
        U[c_idx][r1] += 1; U[d_idx][r1] += 1
        U[a_idx][r2] += 1
        U[d_idx][r3] += 1
        U[a_idx][r4] += 1; U[b_idx][r4] += 1
        U[c_idx][r5] += 1; U[a_idx][r5] -= 1
        U[b_idx][r6] += 1; U[d_idx][r6] -= 1
        
        # V (L_B)
        V[e_idx][r0] += 1; V[h_idx][r0] += 1
        V[e_idx][r1] += 1
        V[f_idx][r2] += 1; V[h_idx][r2] -= 1
        V[g_idx][r3] += 1; V[e_idx][r3] -= 1
        V[h_idx][r4] += 1
        V[e_idx][r5] += 1; V[f_idx][r5] += 1
        V[g_idx][r6] += 1; V[h_idx][r6] += 1
        
        # W (Coefficients)
        # C_00 (i=0, k=k_offset) -> idx k_offset*n + 0
        W[k_offset * n + 0][r0] += 1
        W[k_offset * n + 0][r3] += 1
        W[k_offset * n + 0][r4] -= 1
        W[k_offset * n + 0][r6] += 1
        
        # C_01 (i=0, k=k_offset+1) -> idx (k_offset+1)*n + 0
        W[(k_offset + 1) * n + 0][r2] += 1
        W[(k_offset + 1) * n + 0][r4] += 1
        
        # C_10 (i=1, k=k_offset) -> idx k_offset*n + 1
        W[k_offset * n + 1][r1] += 1
        W[k_offset * n + 1][r3] += 1
        
        # C_11 (i=1, k=k_offset+1) -> idx (k_offset+1)*n + 1
        W[(k_offset + 1) * n + 1][r0] += 1
        W[(k_offset + 1) * n + 1][r1] -= 1
        W[(k_offset + 1) * n + 1][r2] += 1
        W[(k_offset + 1) * n + 1][r5] += 1

    # Block 1: j=0,1, k=0,1
    add_strassen_block(0, 0, 0)
    
    # Block 2: j=0,1, k=2,3
    add_strassen_block(7, 0, 2)
    
    # Block 3: j=0,1, k=4 (2x1)
    # A: a=A_00, b=A_01, c=A_10, d=A_11
    # B: e=B_04, g=B_14
    r0 = 14
    r1 = 15
    r2 = 16
    r3 = 17
    
    U[0 * m + 0][r0] += 1 # a
    U[0 * m + 1][r1] += 1 # b
    U[1 * m + 0][r2] += 1 # c
    U[1 * m + 1][r3] += 1 # d
    
    V[0 * p + 4][r0] += 1 # e
    V[1 * p + 4][r1] += 1 # g
    V[0 * p + 4][r2] += 1 # e
    V[1 * p + 4][r3] += 1 # g
    
    # C_04 (i=0, k=4) -> idx 4*2 + 0 = 8
    W[4 * n + 0][r0] += 1
    W[4 * n + 0][r1] += 1
    
    # C_14 (i=1, k=4) -> idx 4*2 + 1 = 9
    W[4 * n + 1][r2] += 1
    W[4 * n + 1][r3] += 1
    
    # Block 4: j=2, k=0..4 (1x5 outer product)
    # A: u0=A_02, u1=A_12
    # B: v_k=B_2k
    rank_offset = 18
    for k in range(5):
        r_u0 = rank_offset
        r_u1 = rank_offset + 1
        rank_offset += 2
        
        U[0 * m + 2][r_u0] += 1 # u0
        U[1 * m + 2][r_u1] += 1 # u1
        
        V[2 * p + k][r_u0] += 1 # v_k
        V[2 * p + k][r_u1] += 1 # v_k
        
        # C_0k (i=0, k=k) -> idx k*2 + 0
        W[k * n + 0][r_u0] += 1
        # C_1k (i=1, k=k) -> idx k*2 + 1
        W[k * n + 1][r_u1] += 1
        
    return {"U": U, "V": V, "W": W}
