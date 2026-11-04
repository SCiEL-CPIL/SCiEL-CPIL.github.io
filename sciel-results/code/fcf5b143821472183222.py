import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # Helper to create Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1 = (a11+a22)(b11+b22)
        # a11 is U[0], a22 is U[3]
        # b11 is V[0], b22 is V[3]
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 1, 1, 1] # c11, c12, c21, c22? No.
        
        # Let's do it systematically.
        # c11 = P1 + P4 - P5 + P7
        # c12 = P3 + P5
        # c21 = P2 + P4
        # c22 = P1 - P2 + P3 + P6
        
        # W index is k*2 + i.
        # W[0] -> k=0, i=0 -> c11
        # W[1] -> k=0, i=1 -> c21
        # W[2] -> k=1, i=0 -> c12
        # W[3] -> k=1, i=1 -> c22
        
        # P1
        U[:, 0] = [1, 0, 0, 1] # a11, a22
        V[:, 0] = [1, 0, 0, 1] # b11, b22
        W[0, 0] = 1 # c11
        W[3, 0] = 1 # c22
        
        # P2 = (a21+a22)b11
        U[:, 1] = [0, 0, 1, 1] # a21, a22
        V[:, 1] = [1, 0, 0, 0] # b11
        W[1, 1] = 1 # c21
        W[3, 1] = -1 # c22
        
        # P3 = a11(b12-b22)
        U[:, 2] = [1, 0, 0, 0] # a11
        V[:, 2] = [0, 1, 0, -1] # b12, b22
        W[2, 2] = 1 # c12
        W[3, 2] = 1 # c22
        
        # P4 = a22(b21-b11)
        U[:, 3] = [0, 0, 0, 1] # a22
        V[:, 3] = [-1, 0, 1, 0] # b11, b21
        W[0, 3] = 1 # c11
        W[1, 3] = 1 # c21
        
        # P5 = (a11+a12)b22
        U[:, 4] = [1, 1, 0, 0] # a11, a12
        V[:, 4] = [0, 0, 0, 1] # b22
        W[0, 4] = -1 # c11
        W[2, 4] = 1 # c12
        
        # P6 = (a21-a11)(b11+b12)
        U[:, 5] = [-1, 0, 1, 0] # a11, a21
        V[:, 5] = [1, 1, 0, 0] # b11, b12
        W[3, 5] = 1 # c22
        
        # P7 = (a12-a22)(b21+b22)
        U[:, 6] = [0, 1, 0, -1] # a12, a22
        V[:, 6] = [0, 0, 1, 1] # b21, b22
        W[0, 6] = 1 # c11
        W[2, 6] = 1 # c12
        
        return U, V, W

    # Helper to create 2x2x4 decomposition (two 2x2x2 Strassen)
    def strassen_2x2x4():
        U1, V1, W1 = strassen_2x2x2()
        U2, V2, W2 = strassen_2x2x2()
        
        # U is same for both (A is shared)
        # But we need to output rank 14.
        # U_total: (4, 14)
        # V_total: (8, 14)
        # W_total: (8, 14)
        
        U_tot = np.zeros((4, 14))
        V_tot = np.zeros((8, 14))
        W_tot = np.zeros((8, 14))
        
        # Block 1: B1 (cols 0,1 of B)
        U_tot[:, 0:7] = U1
        # V1 rows 0..3 correspond to j=0,1; k=0,1
        # In 2x2x4, V index is j*4+k.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 4
        # j=1, k=1 -> 5
        # V1 local index j*2+k.
        # 0->0, 1->1, 2->4, 3->5
        v_map = [0, 1, 4, 5]
        V_tot[v_map, 0:7] = V1
        
        # W1 rows 0..3 correspond to k=0,1; i=0,1
        # In 2x2x4, W index is k*2+i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # W1 local index k*2+i.
        # 0->0, 1->1, 2->2, 3->3
        w_map = [0, 1, 2, 3]
        W_tot[w_map, 0:7] = W1
        
        # Block 2: B2 (cols 2,3 of B)
        U_tot[:, 7:14] = U2
        # V2 local index j*2+k.
        # j=0, k=0 -> 0. Global k=2 -> index 2.
        # j=0, k=1 -> 1. Global k=3 -> index 3.
        # j=1, k=0 -> 2. Global k=2 -> index 6.
        # j=1, k=1 -> 3. Global k=3 -> index 7.
        v_map2 = [2, 3, 6, 7]
        V_tot[v_map2, 7:14] = V2
        
        # W2 local index k*2+i.
        # k=0, i=0 -> 0. Global k=2 -> index 4.
        # k=0, i=1 -> 1. Global k=2 -> index 5.
        # k=1, i=0 -> 2. Global k=3 -> index 6.
        # k=1, i=1 -> 3. Global k=3 -> index 7.
        w_map2 = [4, 5, 6, 7]
        W_tot[w_map2, 7:14] = W2
        
        return U_tot, V_tot, W_tot

    # Helper to create 2x4x8 decomposition (two 2x2x8 Strassen)
    def strassen_2x4x8():
        U1, V1, W1 = strassen_2x2x4() # 2x2x4
        U2, V2, W2 = strassen_2x2x4() # 2x2x4
        
        # We need 2x2x8 first.
        # 2x2x8 is two 2x2x4 blocks.
        # Let's make a helper for 2x2x8.
        
        def strassen_2x2x8():
            Ua, Va, Wa = strassen_2x2x4()
            Ub, Vb, Wb = strassen_2x2x4()
            
            U_tot = np.zeros((4, 28))
            V_tot = np.zeros((16, 28))
            W_tot = np.zeros((16, 28))
            
            # Block A: B cols 0..3
            U_tot[:, 0:14] = Ua
            # V local (2x2x4) index j*4+k.
            # j=0, k=0..3 -> 0..3
            # j=1, k=0..3 -> 4..7
            # Global (2x2x8) index j*8+k.
            # j=0, k=0..3 -> 0..3
            # j=1, k=0..3 -> 8..11
            v_map = [0, 1, 2, 3, 8, 9, 10, 11]
            V_tot[v_map, 0:14] = Va
            
            # W local (2x2x4) index k*2+i.
            # k=0..3, i=0..1.
            # Global (2x2x8) index k*2+i.
            # k=0..3 -> 0..7
            w_map = [0, 1, 2, 3, 4, 5, 6, 7]
            W_tot[w_map, 0:14] = Wa
            
            # Block B: B cols 4..7
            U_tot[:, 14:28] = Ub
            # V local index j*4+k.
            # j=0, k=0..3 -> 0..3. Global k=4..7 -> 4..7.
            # j=1, k=0..3 -> 4..7. Global k=4..7 -> 12..15.
            v_map2 = [4, 5, 6, 7, 12, 13, 14, 15]
            V_tot[v_map2, 14:28] = Vb
            
            # W local index k*2+i.
            # k=0..3 -> 0..7. Global k=4..7 -> 8..15.
            w_map2 = [8, 9, 10, 11, 12, 13, 14, 15]
            W_tot[w_map2, 14:28] = Wb
            
            return U_tot, V_tot, W_tot

        Ua, Va, Wa = strassen_2x2x8()
        Ub, Vb, Wb = strassen_2x2x8()
        
        U_tot = np.zeros((8, 56))
        V_tot = np.zeros((32, 56))
        W_tot = np.zeros((16, 56))
        
        # A1 (rows 0,1 of A)
        U_tot[0:4, 0:28] = Ua
        # V1 (rows 0..15 of B)
        V_tot[0:16, 0:28] = Va
        # W1 (all of C)
        W_tot[:, 0:28] = Wa
        
        # A2 (rows 2,3 of A)
        U_tot[4:8, 28:56] = Ub
        # V2 (rows 16..31 of B)
        V_tot[16:32, 28:56] = Vb
        # W2 (all of C)
        W_tot[:, 28:56] = Wb
        
        return U_tot, V_tot, W_tot

    # Schoolbook 1x4x8
    def schoolbook_1x4x8():
        n_s, m_s, p_s = 1, 4, 8
        rank = n_s * m_s * p_s
        U = np.zeros((n_s * m_s, rank))
        V = np.zeros((m_s * p_s, rank))
        W = np.zeros((p_s * n_s, rank))
        
        col = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U[i * m_s + j, col] = 1
                    V[j * p_s + k, col] = 1
                    W[k * n_s + i, col] = 1
                    col += 1
        return U, V, W

    U1, V1, W1 = schoolbook_1x4x8()
    U2, V2, W2 = strassen_2x4x8()
    
    U_total = np.zeros((12, 88))
    V_total = np.zeros((32, 88))
    W_total = np.zeros((24, 88))
    
    # 1x4x8 part
    # U rows 0..3
    U_total[0:4, 0:32] = U1
    # V rows 0..31
    V_total[0:32, 0:32] = V1
    # W rows 0..7
    W_total[0:8, 0:32] = W1
    
    # 2x4x8 part
    # U rows 4..11
    U_total[4:12, 32:88] = U2
    # V rows 0..31
    V_total[0:32, 32:88] = V2
    # W rows 8..23
    W_total[8:24, 32:88] = W2
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
