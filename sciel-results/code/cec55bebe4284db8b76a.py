import numpy as np

def construct():
    n, m, p = 3, 4, 7
    
    # Dimensions for the blocks
    # A is 3x4, B is 4x7, C is 3x7
    # Split A into AT (2x4) and AB (1x4)
    # Split B into BL (4x4) and BR (4x3)
    # C splits into CT (2x7) and CB (1x7)
    # CT splits into CTL (2x4) and CTR (2x3)
    # CB splits into CBL (1x4) and CBR (1x3)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a trivial decomposition for n x m x p
    def add_trivial(n_l, m_l, p_l, i_offset, j_offset, k_offset):
        # i_offset: offset in global i (0..n-1)
        # j_offset: offset in global j (0..m-1)
        # k_offset: offset in global k (0..p-1)
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    
                    u[(i + i_offset) * m + (j + j_offset)] = 1
                    v[(j + j_offset) * p + (k + k_offset)] = 1
                    w[(k + k_offset) * n + (i + i_offset)] = 1
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)

    # Helper to add Strassen 2x2x2
    def add_strassen_222(i_offset, j_offset, k_offset):
        # A indices: i in 0,1; j in 0,1
        # B indices: j in 0,1; k in 0,1
        # C indices: i in 0,1; k in 0,1
        
        # M1 = (a+d)(e+h)
        # a=A00, b=A01, c=A10, d=A11
        # e=B00, f=B01, g=B10, h=B11
        
        # U vectors (coeffs of A)
        u1 = np.zeros(n*m); u1[(i_offset)*m + j_offset] = 1; u1[(i_offset+1)*m + (j_offset+1)] = 1
        u2 = np.zeros(n*m); u2[(i_offset+1)*m + j_offset] = 1; u2[(i_offset+1)*m + (j_offset+1)] = 1
        u3 = np.zeros(n*m); u3[(i_offset)*m + j_offset] = 1
        u4 = np.zeros(n*m); u4[(i_offset+1)*m + (j_offset+1)] = 1
        u5 = np.zeros(n*m); u5[(i_offset)*m + j_offset] = 1; u5[(i_offset)*m + (j_offset+1)] = 1
        u6 = np.zeros(n*m); u6[(i_offset+1)*m + j_offset] = 1; u6[(i_offset)*m + j_offset] = -1
        u7 = np.zeros(n*m); u7[(i_offset)*m + (j_offset+1)] = 1; u7[(i_offset+1)*m + (j_offset+1)] = -1
        
        # V vectors (coeffs of B)
        v1 = np.zeros(m*p); v1[(j_offset)*p + k_offset] = 1; v1[(j_offset+1)*p + (k_offset+1)] = 1
        v2 = np.zeros(m*p); v2[(j_offset)*p + k_offset] = 1
        v3 = np.zeros(m*p); v3[(j_offset)*p + (k_offset+1)] = 1; v3[(j_offset+1)*p + (k_offset+1)] = -1
        v4 = np.zeros(m*p); v4[(j_offset+1)*p + k_offset] = 1; v4[(j_offset+1)*p + (k_offset+1)] = -1
        v5 = np.zeros(m*p); v5[(j_offset+1)*p + (k_offset+1)] = 1
        v6 = np.zeros(m*p); v6[(j_offset)*p + k_offset] = 1; v6[(j_offset)*p + (k_offset+1)] = 1
        v7 = np.zeros(m*p); v7[(j_offset+1)*p + k_offset] = 1; v7[(j_offset+1)*p + (k_offset+1)] = 1
        
        # W vectors (coeffs of C)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        w1 = np.zeros(p*n); w1[(k_offset)*n + i_offset] = 1; w1[(k_offset+1)*n + (i_offset+1)] = 1
        w2 = np.zeros(p*n); w2[(k_offset)*n + (i_offset+1)] = 1
        w3 = np.zeros(p*n); w3[(k_offset+1)*n + i_offset] = 1
        w4 = np.zeros(p*n); w4[(k_offset)*n + (i_offset+1)] = 1; w4[(k_offset+1)*n + (i_offset+1)] = 1
        w5 = np.zeros(p*n); w5[(k_offset+1)*n + i_offset] = -1
        w6 = np.zeros(p*n); w6[(k_offset+1)*n + (i_offset+1)] = 1
        w7 = np.zeros(p*n); w7[(k_offset)*n + i_offset] = 1
        
        U_cols.extend([u1, u2, u3, u4, u5, u6, u7])
        V_cols.extend([v1, v2, v3, v4, v5, v6, v7])
        W_cols.extend([w1, w2, w3, w4, w5, w6, w7])

    # Helper to add 2x2x3 decomposition (Strassen 2x2x2 + Trivial 2x2x1)
    def add_223(i_offset, j_offset, k_offset):
        # A: 2x2, B: 2x3, C: 2x3
        # Split B into B1 (2x2) and B2 (2x1)
        # B1 cols k_offset, k_offset+1
        # B2 col k_offset+2
        
        # Strassen for A * B1
        add_strassen_222(i_offset, j_offset, k_offset)
        
        # Trivial for A * B2
        # A is 2x2, B2 is 2x1, C2 is 2x1
        # k2 = k_offset + 2
        for i in range(2):
            for j in range(2):
                u = np.zeros(n*m)
                u[(i + i_offset)*m + (j + j_offset)] = 1
                
                v = np.zeros(m*p)
                v[(j + j_offset)*p + (k_offset + 2)] = 1
                
                w = np.zeros(p*n)
                w[(k_offset + 2)*n + (i + i_offset)] = 1
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)

    # Block 1: CTL (2x4x4)
    # A_T (2x4) = [A_T1 (2x2), A_T2 (2x2)]
    # B_L (4x4) = [B_L1 (2x4); B_L2 (2x4)]
    # CTL = A_T1 B_L1 + A_T2 B_L2
    
    # Sub-block 1a: A_T1 (2x2) * B_L1 (2x4)
    # A_T1: i=0,1; j=0,1
    # B_L1: j=0,1; k=0,1,2,3
    # This is 2x2x4.
    # Split B_L1 into B_L1a (2x2) and B_L1b (2x2)
    # B_L1a: k=0,1
    # B_L1b: k=2,3
    
    # 1a-i: 2x2x2 (Strassen)
    add_strassen_222(0, 0, 0)
    # 1a-ii: 2x2x2 (Strassen)
    add_strassen_222(0, 0, 2)
    
    # Sub-block 1b: A_T2 (2x2) * B_L2 (2x4)
    # A_T2: i=0,1; j=2,3
    # B_L2: j=2,3; k=0,1,2,3
    # Split B_L2 into B_L2a (2x2) and B_L2b (2x2)
    # B_L2a: k=0,1
    # B_L2b: k=2,3
    
    # 1b-i: 2x2x2 (Strassen)
    add_strassen_222(0, 2, 0)
    # 1b-ii: 2x2x2 (Strassen)
    add_strassen_222(0, 2, 2)
    
    # Block 2: CTR (2x4x3)
    # A_T (2x4) = [A_T1, A_T2]
    # B_R (4x3) = [B_R1 (2x3); B_R2 (2x3)]
    # CTR = A_T1 B_R1 + A_T2 B_R2
    
    # Sub-block 2a: A_T1 (2x2) * B_R1 (2x3)
    # A_T1: i=0,1; j=0,1
    # B_R1: j=0,1; k=4,5,6
    # Use 2x2x3 decomposition
    add_223(0, 0, 4)
    
    # Sub-block 2b: A_T2 (2x2) * B_R2 (2x3)
    # A_T2: i=0,1; j=2,3
    # B_R2: j=2,3; k=4,5,6
    # Use 2x2x3 decomposition
    add_223(0, 2, 4)
    
    # Block 3: CBL (1x4x4)
    # A_B (1x4)
    # B_L (4x4)
    # Trivial
    add_trivial(1, 4, 4, 2, 0, 0)
    
    # Block 4: CBR (1x4x3)
    # A_B (1x4)
    # B_R (4x3)
    # Trivial
    add_trivial(1, 4, 3, 2, 0, 4)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
