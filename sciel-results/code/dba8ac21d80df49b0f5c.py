import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen(a_idx, b_idx, c_idx):
        # a_idx, b_idx, c_idx are lists of 4 indices [00, 01, 10, 11]
        # M1 = (a00+a11)(b00+b11)
        u1 = [0]*(n*m); u1[a_idx[0]] = 1; u1[a_idx[3]] = 1
        v1 = [0]*(m*p); v1[b_idx[0]] = 1; v1[b_idx[3]] = 1
        w1 = [0]*(p*n)
        w1[c_idx[0]] += 1
        w1[c_idx[3]] += 1
        add_term(u1, v1, w1)

        # M2 = (a10+a11)b00
        u2 = [0]*(n*m); u2[a_idx[2]] = 1; u2[a_idx[3]] = 1
        v2 = [0]*(m*p); v2[b_idx[0]] = 1
        w2 = [0]*(p*n)
        w2[c_idx[2]] += 1
        w2[c_idx[3]] -= 1
        add_term(u2, v2, w2)

        # M3 = a00(b01-b11)
        u3 = [0]*(n*m); u3[a_idx[0]] = 1
        v3 = [0]*(m*p); v3[b_idx[1]] = 1; v3[b_idx[3]] = -1
        w3 = [0]*(p*n)
        w3[c_idx[1]] += 1
        w3[c_idx[3]] += 1
        add_term(u3, v3, w3)

        # M4 = a11(b10-b00)
        u4 = [0]*(n*m); u4[a_idx[3]] = 1
        v4 = [0]*(m*p); v4[b_idx[2]] = 1; v4[b_idx[0]] = -1
        w4 = [0]*(p*n)
        w4[c_idx[0]] += 1
        w4[c_idx[2]] += 1
        add_term(u4, v4, w4)

        # M5 = (a00+a01)b11
        u5 = [0]*(n*m); u5[a_idx[0]] = 1; u5[a_idx[1]] = 1
        v5 = [0]*(m*p); v5[b_idx[3]] = 1
        w5 = [0]*(p*n)
        w5[c_idx[0]] -= 1
        w5[c_idx[1]] += 1
        add_term(u5, v5, w5)

        # M6 = (a10-a00)(b00+b01)
        u6 = [0]*(n*m); u6[a_idx[2]] = 1; u6[a_idx[0]] = -1
        v6 = [0]*(m*p); v6[b_idx[0]] = 1; v6[b_idx[1]] = 1
        w6 = [0]*(p*n)
        w6[c_idx[3]] += 1
        add_term(u6, v6, w6)

        # M7 = (a01-a11)(b10+b11)
        u7 = [0]*(n*m); u7[a_idx[1]] = 1; u7[a_idx[3]] = -1
        v7 = [0]*(m*p); v7[b_idx[2]] = 1; v7[b_idx[3]] = 1
        w7 = [0]*(p*n)
        w7[c_idx[0]] += 1
        add_term(u7, v7, w7)

    # Part 1: A[0:2] * B[0:4]
    # A_L = A[0:2, 0:2] -> U indices: 0, 1, 4, 5
    # A_R = A[0:2, 2:4] -> U indices: 2, 3, 6, 7
    
    # B_LL = B[0:2, 0:2] -> V indices: 0, 1, 8, 9
    # B_LR = B[0:2, 2:4] -> V indices: 2, 3, 10, 11
    # B_RL = B[2:4, 0:2] -> V indices: 16, 17, 24, 25
    # B_RR = B[2:4, 2:4] -> V indices: 18, 19, 26, 27
    
    # C[0:2, 0:2] W indices: 0, 1, 3, 4
    # C[0:2, 2:4] W indices: 2, 5, 6, 7
    
    A_L_u = [0, 1, 4, 5]
    A_R_u = [2, 3, 6, 7]
    
    B_LL_v = [0, 1, 8, 9]
    B_LR_v = [2, 3, 10, 11]
    B_RL_v = [16, 17, 24, 25]
    B_RR_v = [18, 19, 26, 27]
    
    C_TL_w = [0, 1, 3, 4]
    C_TR_w = [2, 5, 6, 7]
    
    add_strassen(A_L_u, B_LL_v, C_TL_w)
    add_strassen(A_R_u, B_RL_v, C_TL_w)
    add_strassen(A_L_u, B_LR_v, C_TR_w)
    add_strassen(A_R_u, B_RR_v, C_TR_w)

    # Part 2: A[0:2] * B[4:8]
    # B_RL2 = B[0:2, 4:6] -> V indices: 4, 5, 12, 13
    # B_RR2 = B[0:2, 6:8] -> V indices: 6, 7, 14, 15
    # B_RL3 = B[2:4, 4:6] -> V indices: 20, 21, 28, 29
    # B_RR3 = B[2:4, 6:8] -> V indices: 22, 23, 30, 31
    
    # C[0:2, 4:6] W indices: 4, 7, 9, 10
    # C[0:2, 6:8] W indices: 6, 11, 12, 13
    
    B_RL2_v = [4, 5, 12, 13]
    B_RR2_v = [6, 7, 14, 15]
    B_RL3_v = [20, 21, 28, 29]
    B_RR3_v = [22, 23, 30, 31]
    
    C_BL_w = [4, 7, 9, 10]
    C_BR_w = [6, 11, 12, 13]
    
    add_strassen(A_L_u, B_RL2_v, C_BL_w)
    add_strassen(A_R_u, B_RL3_v, C_BL_w)
    add_strassen(A_L_u, B_RR2_v, C_BR_w)
    add_strassen(A_R_u, B_RR3_v, C_BR_w)

    # Part 3: A[2] * B
    # Rank 32 terms
    for k in range(4):
        for j in range(8):
            u = [0]*(n*m); u[2*4 + k] = 1
            v = [0]*(m*p); v[k*8 + j] = 1
            w = [0]*(p*n); w[j*3 + 2] = 1
            add_term(u, v, w)

    # Convert columns to rows
    rank = len(U_cols)
    if rank == 0:
        return {"U": [], "V": [], "W": []}
    
    U_mat = [list(row) for row in zip(*U_cols)]
    V_mat = [list(row) for row in zip(*V_cols)]
    W_mat = [list(row) for row in zip(*W_cols)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
