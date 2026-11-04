def construct():
    n, m, p = N, M, P
    U = np.zeros((15, 61))
    V = np.zeros((35, 61))
    W = np.zeros((21, 61))
    
    col_offset = 0
    
    def add_decomposition(U_l, V_l, W_l, u_rows, v_rows, w_rows):
        nonlocal col_offset
        rank_l = U_l.shape[1]
        for r in range(rank_l):
            for local_row, global_row in zip(u_rows, u_rows): # Wait, u_rows should be global indices
                pass
                
    # Better:
    def add_decomp(U_l, V_l, W_l, u_global_indices, v_global_indices, w_global_indices):
        nonlocal col_offset
        rank_l = U_l.shape[1]
        for r in range(rank_l):
            for local_r, global_r in enumerate(u_global_indices):
                U[global_r, col_offset + r] += U_l[local_r, r]
            for local_r, global_r in enumerate(v_global_indices):
                V[global_r, col_offset + r] += V_l[local_r, r]
            for local_r, global_r in enumerate(w_global_indices):
                W[global_r, col_offset + r] += W_l[local_r, r]
        col_offset += rank_l

    # 1. A4L (3x2) * B4L (2x7)
    # u_global: 3x2 block at j=0,1. Indices i*5+j.
    u_g_4L = [i*5+j for i in range(3) for j in range(2)]
    
    # 1a. B4L1 (2x3) at k=0,1,2.
    # v_global: 2x3 block at j=0,1, k=0,1,2. Indices j*7+k.
    v_g_4L1 = [j*7+k for j in range(2) for k in range(3)]
    # w_global: 3x3 block at k=0,1,2. Indices k*3+i.
    w_g_4L1 = [k*3+i for k in range(3) for i in range(3)]
    
    U_l, V_l, W_l = laderman_3x2x3()
    add_decomp(U_l, V_l, W_l, u_g_4L, v_g_4L1, w_g_4L1)
    
    # 1b. B4L2 (2x3) at k=3,4,5.
    v_g_4L2 = [j*7+k for j in range(2) for k in range(3, 6)]
    w_g_4L2 = [k*3+i for k in range(3, 6) for i in range(3)]
    
    U_l, V_l, W_l = laderman_3x2x3()
    add_decomp(U_l, V_l, W_l, u_g_4L, v_g_4L2, w_g_4L2)
    
    # 1c. B4L3 (2x1) at k=6.
    U_s, V_s, W_s = schoolbook_3x2x1()
    v_g_4L3 = [j*7+6 for j in range(2)]
    w_g_4L3 = [6*3+i for i in range(3)]
    
    add_decomp(U_s, V_s, W_s, u_g_4L, v_g_4L3, w_g_4L3)
    
    # 2. A4R (3x2) * B4R (2x7)
    # u_global: 3x2 block at j=2,3. Indices i*5+j.
    u_g_4R = [i*5+j for i in range(3) for j in range(2, 4)]
    
    # 2a. B4R1 (2x3) at k=0,1,2.
    v_g_4R1 = [j*7+k for j in range(2, 4) for k in range(3)]
    w_g_4R1 = [k*3+i for k in range(3) for i in range(3)]
    
    U_l, V_l, W_l = laderman_3x2x3()
    add_decomp(U_l, V_l, W_l, u_g_4R, v_g_4R1, w_g_4R1)
    
    # 2b. B4R2 (2x3) at k=3,4,5.
    v_g_4R2 = [j*7+k for j in range(2, 4) for k in range(3, 6)]
    w_g_4R2 = [k*3+i for k in range(3, 6) for i in range(3)]
    
    U_l, V_l, W_l = laderman_3x2x3()
    add_decomp(U_l, V_l, W_l, u_g_4R, v_g_4R2, w_g_4R2)
    
    # 2c. B4R3 (2x1) at k=6.
    U_s, V_s, W_s = schoolbook_3x2x1()
    v_g_4R3 = [j*7+6 for j in range(2, 4)]
    w_g_4R3 = [6*3+i for i in range(3)]
    
    add_decomp(U_s, V_s, W_s, u_g_4R, v_g_4R3, w_g_4R3)
    
    # 3. A1 (3x1) * B1 (1x7)
    # u_global: 3x1 block at j=4. Indices i*5+4.
    u_g_1 = [i*5+4 for i in range(3)]
    
    # 3a. B1 (1x7) at k=0..6.
    U_s, V_s, W_s = schoolbook_3x1x7()
    v_g_1 = [4*7+k for k in range(7)]
    w_g_1 = [k*3+i for k in range(7) for i in range(3)]
    
    add_decomp(U_s, V_s, W_s, u_g_1, v_g_1, w_g_1)
    
    if not check_decomposition(U.tolist(), V.tolist(), W.tolist()):
         # Fallback
         U_sb, V_sb, W_sb = schoolbook_nm_p(n, m, p)
         return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
         
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
