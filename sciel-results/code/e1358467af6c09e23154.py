def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 8 x 28
    
    # Strassen decomposition for 2x2x2
    # Returns (U_2x2, V_2x2, W_2x2) for computing X*Y = Z where X,Y,Z are 2x2
    # U has shape (4,7), V has shape (4,7), W has shape (4,7)
    # Indices: U[x_row*2+x_col, s], V[y_row*2+y_col, s], W[z_row*2+z_col, s]
    
    def strassen_2x2():
        # Standard Strassen
        # M1 = X00*(Y00+Y11)
        # M2 = (X10+X11)*Y00
        # M3 = (X00+X01)*Y11
        # M4 = X11*(Y10+Y00)
        # M5 = (X00+X10)*(Y01+Y10)
        # M6 = (X11+X01)*(Y10+Y11)
        # M7 = (X01+X11)*(Y00-Y10)
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M6
        # Z10 = M2+M5
        # Z11 = M1+M3-M2+M6
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # X indices: 0=X00, 1=X01, 2=X10, 3=X11
        # Y indices: 0=Y00, 1=Y01, 2=Y10, 3=Y11
        # Z indices: 0=Z00, 1=Z01, 2=Z10, 3=Z11
        
        # M1: X00 * (Y00+Y11)
        U_s[0][0] = 1
        V_s[0][0] = 1
        V_s[3][0] = 1
        # Z00 += M1, Z11 += M1
        W_s[0][0] = 1
        W_s[3][0] = 1
        
        # M2: (X10+X11) * Y00
        U_s[2][1] = 1
        U_s[3][1] = 1
        V_s[0][1] = 1
        # Z10 += M2
        W_s[2][1] = 1
        
        # M3: (X00+X01) * Y11
        U_s[0][2] = 1
        U_s[1][2] = 1
        V_s[3][2] = 1
        # Z01 += M3, Z11 += M3
        W_s[1][2] = 1
        W_s[3][2] = 1
        
        # M4: X11 * (Y10+Y00)
        U_s[3][3] = 1
        V_s[2][3] = 1
        V_s[0][3] = 1
        # Z00 += M4
        W_s[0][3] = 1
        
        # M5: (X00+X10) * (Y01+Y10)
        U_s[0][4] = 1
        U_s[2][4] = 1
        V_s[1][4] = 1
        V_s[2][4] = 1
        # Z00 -= M5
        W_s[0][4] = -1
        
        # M6: (X11+X01) * (Y10+Y11)
        U_s[3][5] = 1
        U_s[1][5] = 1
        V_s[2][5] = 1
        V_s[3][5] = 1
        # Z01 += M6, Z11 -= M6
        W_s[1][5] = 1
        W_s[3][5] = -1
        
        # M7: (X01+X11) * (Y00-Y10)
        U_s[1][6] = 1
        U_s[3][6] = 1
        V_s[0][6] = 1
        V_s[2][6] = -1
        # Z00 += M7, Z11 += M7... wait let me recheck
        # Z11 = M1+M3-M2+M6, so M7 doesn't appear in Z11
        # Z00 = M1+M4-M5+M7
        W_s[0][6] = 1
        
        return U_s, V_s, W_s
    
    # Now build the 2x4x4 decomposition
    # Split j into {0,1} (top) and {2,3} (bottom)
    # For top half: A_top is 2x2 (i=0,1; j=0,1), B_top is 2x4 (j=0,1; k=0,1,2,3)
    # Compute C_top = A_top * B_top using Strassen on 2x2x2 blocks
    
    # For B_top, split columns into (0,1) and (2,3)
    # Use Strassen for each 2x2x2 sub-problem
    
    U_s, V_s, W_s = strassen_2x2()
    
    # Top half, columns 0,1:
    # A indices: i*2+j where i=0,1, j=0,1 -> A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: j*4+k where j=0,1, k=0,1 -> B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C indices: k*2+i where k=0,1, i=0,1 -> C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
    
    for s in range(7):
        # U: A indices
        for a_idx in range(4):
            if U_s[a_idx][s] != 0:
                # a_idx maps to A[i,j]: a_idx = i*2+j
                U[a_idx][s] += U_s[a_idx][s]
        
        # V: B indices for j=0,1, k=0,1
        # V_s index: b_idx = j*2+k for j=0,1, k=0,1
        # In full V: index = j*4+k
        for b_idx in range(4):
            if V_s[b_idx][s] != 0:
                j = b_idx // 2
                k = b_idx % 2
                full_b = j * 4 + k
                V[full_b][s] += V_s[b_idx][s]
        
        # W: C indices for i=0,1, k=0,1
        # W_s index: c_idx = i*2+k for i=0,1, k=0,1
        # In full W: index = k*2+i (trace convention W[k*n+i,r])
        for c_idx in range(4):
            if W_s[c_idx][s] != 0:
                i = c_idx // 2
                k = c_idx % 2
                full_c = k * 2 + i
                W[full_c][s] += W_s[c_idx][s]
    
    # Top half, columns 2,3:
    for s in range(7):
        base = 7
        for a_idx in range(4):
            if U_s[a_idx][s] != 0:
                U[a_idx][base + s] += U_s[a_idx][s]
        
        for b_idx in range(4):
            if V_s[b_idx][s] != 0:
                j = b_idx // 2
                k = b_idx % 2
                full_b = j * 4 + (2 + k)  # k=2,3
                V[full_b][base + s] += V_s[b_idx][s]
        
        for c_idx in range(4):
            if W_s[c_idx][s] != 0:
                i = c_idx // 2
                k = c_idx % 2
                full_c = (2 + k) * 2 + i
                W[full_c][base + s] += W_s[c_idx][s]
    
    # Bottom half: j=2,3
    # A indices: i*2+j where i=0,1, j=2,3 -> A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7
    # B indices: j*4+k where j=2,3, k=0,1,2,3
    # C indices: same as top
    
    base = 14
    # Bottom half, columns 0,1:
    for s in range(7):
        for a_idx in range(4):
            if U_s[a_idx][s] != 0:
                # a_idx = i*2+j_local where j_local=0,1 maps to j=2,3
                i = a_idx // 2
                j_local = a_idx % 2
                full_a = i * 4 + (2 + j_local)
                U[full_a][base + s] += U_s[a_idx][s]
        
        for b_idx in range(4):
            if V_s[b_idx][s] != 0:
                j_local = b_idx // 2
                k = b_idx % 2
                full_b = (2 + j_local) * 4 + k
                V[full_b][base + s] += V_s[b_idx][s]
        
        for c_idx in range(4):
            if W_s[c_idx][s] != 0:
                i = c_idx // 2
                k = c_idx % 2
                full_c = k * 2 + i
                W[full_c][base + s] += W_s[c_idx][s]
    
    # Bottom half, columns 2,3:
    base2 = 21
    for s in range(7):
        for a_idx in range(4):
            if U_s[a_idx][s] != 0:
                i = a_idx // 2
                j_local = a_idx % 2
                full_a = i * 4 + (2 + j_local)
                U[full_a][base2 + s] += U_s[a_idx][s]
        
        for b_idx in range(4):
            if V_s[b_idx][s] != 0:
                j_local = b_idx // 2
                k = b_idx % 2
                full_b = (2 + j_local) * 4 + (2 + k)
                V[full_b][base2 + s] += V_s[b_idx][s]
        
        for c_idx in range(4):
            if W_s[c_idx][s] != 0:
                i = c_idx // 2
                k = c_idx % 2
                full_c = (2 + k) * 2 + i
                W[full_c][base2 + s] += W_s[c_idx][s]
    
    return {"U": U, "V": V, "W": W}
