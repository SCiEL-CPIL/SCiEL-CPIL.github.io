def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients
    # r: 0..6
    # U[i, j, r]
    U_c = np.zeros((2, 2, 7))
    V_c = np.zeros((2, 2, 7))
    W_c = np.zeros((2, 2, 7))
    
    # M1
    U_c[0,0,0] = 1; U_c[1,1,0] = 1
    V_c[0,0,0] = 1; V_c[1,1,0] = 1
    W_c[0,0,0] = 1; W_c[0,1,0] = 0; W_c[1,0,0] = 0; W_c[1,1,0] = 1 # z00, z11
    
    # M2
    U_c[1,0,1] = 1; U_c[1,1,1] = 1
    V_c[0,0,1] = 1
    W_c[0,1,1] = 1 # z10
    W_c[1,1,1] = -1 # z11
    
    # M3
    U_c[0,0,2] = 1
    V_c[0,1,2] = 1; V_c[1,1,2] = -1
    W_c[1,0,2] = 1 # z01
    W_c[1,1,2] = 1 # z11
    
    # M4
    U_c[1,1,3] = 1
    V_c[1,0,3] = 1; V_c[0,0,3] = -1
    W_c[0,0,3] = 1 # z00
    W_c[0,1,3] = 1 # z10
    
    # M5
    U_c[0,0,4] = 1; U_c[0,1,4] = 1
    V_c[1,1,4] = 1
    W_c[0,0,4] = -1 # z00
    W_c[1,0,4] = 1 # z01
    
    # M6
    U_c[1,0,5] = 1; U_c[0,0,5] = -1
    V_c[0,0,5] = 1; V_c[0,1,5] = 1
    W_c[1,1,5] = 1 # z11
    
    # M7
    U_c[0,1,6] = 1; U_c[1,1,6] = -1
    V_c[1,0,6] = 1; V_c[1,1,6] = 1
    W_c[0,0,6] = 1 # z00
    
    blocks = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 2)
    ]
    
    col = 0
    for (i_s, j_s_A, j_s_B, k_s) in blocks:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (i_s + i) * m + (j_s_A + j)
                    U[u_idx, col] = U_c[i, j, r]
            for j in range(2):
                for k in range(2):
                    v_idx = (j_s_B + j) * p + (k_s + k)
                    V[v_idx, col] = V_c[j, k, r]
            for k in range(2):
                for i in range(2):
                    w_idx = (k_s + k) * n + (i_s + i)
                    W[w_idx, col] = W_c[k, i, r]
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
