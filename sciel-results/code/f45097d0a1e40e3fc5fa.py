def _build_strassen_blocks_v2():
    """Build decomposition using 2x2x3 Strassen blocks."""
    U_s23, V_s23, W_s23 = _strassen_2x2x3()
    U_s22, V_s22, W_s22 = _strassen_2x2x2()
    
    all_U = []
    all_V = []
    all_W = []
    
    # Block 1: i_off=0, j_off=0, k_off=0, 2x2x3
    U1, V1, W1 = _embed_strassen_2x2x3(U_s23, V_s23, W_s23, 0, 0, 0)
    all_U.append(U1); all_V.append(V1); all_W.append(W1)
    
    # Block 2: i_off=0, j_off=0, k_off=3, 2x2x2
    U2, V2, W2 = _embed_strassen(U_s22, V_s22, W_s22, 0, 0, 3)
    all_U.append(U2); all_V.append(V2); all_W.append(W2)
    
    # Block 3: i_off=0, j_off=2, k_off=0, 2x2x3
    U3, V3, W3 = _embed_strassen_2x2x3(U_s23, V_s23, W_s23, 0, 2, 0)
    all_U.append(U3); all_V.append(V3); all_W.append(W3)
    
    # Block 4: i_off=0, j_off=2, k_off=3, 2x2x2
    U4, V4, W4 = _embed_strassen(U_s22, V_s22, W_s22, 0, 2, 3)
    all_U.append(U4); all_V.append(V4); all_W.append(W4)
    
    # Block 5: i=2, j_off=0, 1x2x5 schoolbook
    for lj in range(2):
        for gk in range(5):
            U_col = np.zeros(N*M)
            V_col = np.zeros(M*P)
            W_col = np.zeros(P*N)
            U_col[2*M + (lj+0)] = 1.0
            V_col[(lj+0)*P + gk] = 1.0
            W_col[gk*N + 2] = 1.0
            all_U.append(U_col.reshape(-1,1)); all_V.append(V_col.reshape(-1,1)); all_W.append(W_col.reshape(-1,1))
    
    # Block 6: i=2, j_off=2, 1x2x5 schoolbook
    for lj in range(2):
        for gk in range(5):
            U_col = np.zeros(N*M)
            V_col = np.zeros(M*P)
            W_col = np.zeros(P*N)
            U_col[2*M + (lj+2)] = 1.0
            V_col[(lj+2)*P + gk] = 1.0
            W_col[gk*N + 2] = 1.0
            all_U.append(U_col.reshape(-1,1)); all_V.append(V_col.reshape(-1,1)); all_W.append(W_col.reshape(-1,1))
    
    U_final = np.hstack(all_U)
    V_final = np.hstack(all_V)
    W_final = np.hstack(all_W)
    
    return U_final, V_final, W_final, U_final.shape[1]
