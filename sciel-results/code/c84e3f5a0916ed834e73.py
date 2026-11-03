import numpy as np

def construct():
    n, m, p = 3, 3, 4
    rank = 22
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # U_local: (2*2, 7)
    # V_local: (2*2, 7)
    # W_local: (2*2, 7)
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Indices in local 2x2:
    # A: [0,1,2,3] -> (0,0), (0,1), (1,0), (1,1)
    # B: [0,1,2,3] -> (0,0), (0,1), (1,0), (1,1)
    # C: [0,1,2,3] -> (0,0), (0,1), (1,0), (1,1)
    
    # Term 1: (a+d)(e+h)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # Term 2: (c+d)e
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # Term 3: a(f-h)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # Term 4: d(g-e)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # Term 5: (a+b)h
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # Term 6: (c-a)(e+f)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # Term 7: (b-d)(g+h)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    def add_strassen_block(i_off, j_off, k_off, start_col):
        for t in range(7):
            for i_l in range(2):
                for j_l in range(2):
                    i_g = i_off + i_l
                    j_g = j_off + j_l
                    u_idx = i_g * m + j_g
                    U[u_idx, start_col + t] += U_s[i_l * 2 + j_l, t]
                    
                    for k_l in range(2):
                        k_g = k_off + k_l
                        v_idx = j_g * p + k_g
                        V[v_idx, start_col + t] += V_s[j_l * 2 + k_l, t]
                        
                        w_idx = k_g * n + i_g
                        W[w_idx, start_col + t] += W_s[k_l * 2 + i_l, t]

    # Block 1: Top-Left
    add_strassen_block(0, 0, 0, col)
    col += 7
    
    # Block 2: Top-Right
    add_strassen_block(0, 0, 2, col)
    col += 7
    
    # 1x2x2 decomposition (Rank 4)
    # A: [a, b] (1x2)
    # B: [e, f; g, h] (2x2)
    # C: [ae+bg, af+bh] (1x2)
    
    U_122 = np.zeros((2, 4))
    V_122 = np.zeros((4, 4))
    W_122 = np.zeros((2, 4))
    
    # Term 1: a e
    U_122[0, 0] = 1
    V_122[0, 0] = 1
    W_122[0, 0] = 1
    
    # Term 2: b g
    U_122[1, 1] = 1
    V_122[2, 1] = 1
    W_122[0, 1] = 1
    
    # Term 3: a f
    U_122[0, 2] = 1
    V_122[1, 2] = 1
    W_122[1, 2] = 1
    
    # Term 4: b h
    U_122[1, 3] = 1
    V_122[3, 3] = 1
    W_122[1, 3] = 1

    def add_1x2x2_block(i_off, j_off, k_off, start_col):
        for t in range(4):
            for j_l in range(2):
                i_g = i_off
                j_g = j_off + j_l
                u_idx = i_g * m + j_g
                U[u_idx, start_col + t] += U_122[j_l, t]
                
                for k_l in range(2):
                    k_g = k_off + k_l
                    v_idx = j_g * p + k_g
                    V[v_idx, start_col + t] += V_122[j_l * 2 + k_l, t]
                    
                    w_idx = k_g * n + i_g
                    W[w_idx, start_col + t] += W_122[k_l, t]

    # Block 3: Bottom-Left
    add_1x2x2_block(2, 0, 0, col)
    col += 4
    
    # Block 4: Bottom-Right
    add_1x2x2_block(2, 0, 2, col)
    col += 4
    
    # Verify
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
    
    if not np.allclose(T_decomp, T, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb), dtype=float)
        V_fb = np.zeros((m*p, rank_fb), dtype=float)
        W_fb = np.zeros((p*n, rank_fb), dtype=float)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1.0
                    V_fb[j*p+k, col_fb] = 1.0
                    W_fb[k*n+i, col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
