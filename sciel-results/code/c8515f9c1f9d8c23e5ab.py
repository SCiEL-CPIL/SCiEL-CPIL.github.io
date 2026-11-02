# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    
    Strategy: Split the middle dimension m=3 into 2+1.
    Split the output dimension p=7 into 2+2+3.
    
    Blocks:
    1. A[:, 0:2] * B[0:2, 0:2] (2x2x2) -> Strassen (rank 7)
    2. A[:, 0:2] * B[0:2, 2:4] (2x2x2) -> Strassen (rank 7)
    3. A[:, 0:2] * B[0:2, 4:7] (2x2x3) -> Split into 2x2x2 (Strassen, rank 7) + 2x2x1 (Schoolbook, rank 4)
    4. A[:, 2] * B[2, 0:7] (2x1x7) -> Schoolbook (rank 14)
    
    Total Rank: 7 + 7 + (7 + 4) + 14 = 39.
    """
    n, m, p = 2, 3, 7
    
    U = []
    V = []
    W = []
    
    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        # Adds 7 terms for the 2x2x2 block starting at (i_offset, j_offset, k_offset)
        # i in {0,1}, j in {0,1}, k in {0,1}
        # Strassen factors:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the columns for U, V, W
        # U rows: (i, j)
        # V rows: (j, k)
        # W rows: (k, i)
        
        # M1
        U.append([1, 0, 0, 1, 0, 0]) # A00, A01, A10, A11, A20, A21 (local indices 0,1,2,3,4,5)
        # Wait, U has global shape (6, R). We are appending to a list of columns.
        # It is easier to build the columns as lists of length 6, 21, 14.
        
        # Let's restructure to build columns directly.
        pass

    # Better approach: Build lists of columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen_2x2x2(j_start, k_start):
        # i is always 0,1 for the A part (rows 0,1)
        # j is j_start, j_start+1
        # k is k_start, k_start+1
        
        # Local indices in U (6x), V (21x), W (14x)
        # U[i*3 + j]
        # V[j*7 + k]
        # W[k*2 + i]
        
        # M1: (A00+A11)(B00+B11)
        # A part: A[0, j_start] + A[1, j_start+1]
        # B part: B[j_start, k_start] + B[j_start+1, k_start+1]
        # W part: C[0, k_start] += 1, C[1, k_start+1] += 1
        # Wait, C indices: C[i, k].
        # M1 contributes to C[0, k_start] and C[1, k_start+1]?
        # No, Strassen computes the full 2x2 product.
        # C[0, k_start] gets M1.
        # C[0, k_start+1] gets M3+M5.
        # C[1, k_start] gets M2+M4.
        # C[1, k_start+1] gets M1-M2+M3+M6.
        
        # So W vectors must distribute the coefficients to the correct (k, i) slots.
        
        # M1
        u1 = [0]*6
        u1[0*3 + j_start] = 1
        u1[1*3 + (j_start+1)] = 1
        v1 = [0]*21
        v1[j_start*7 + k_start] = 1
        v1[(j_start+1)*7 + (k_start+1)] = 1
        w1 = [0]*14
        w1[k_start*2 + 0] = 1
        w1[(k_start+1)*2 + 1] = 1
        add_term(u1, v1, w1)
        
        # M2: (A10+A11)B00
        u2 = [0]*6
        u2[1*3 + j_start] = 1
        u2[1*3 + (j_start+1)] = 1
        v2 = [0]*21
        v2[j_start*7 + k_start] = 1
        w2 = [0]*14
        w2[k_start*2 + 1] = 1
        add_term(u2, v2, w2)
        
        # M3: A00(B01-B11)
        u3 = [0]*6
        u3[0*3 + j_start] = 1
        v3 = [0]*21
        v3[j_start*7 + (k_start+1)] = 1
        v3[(j_start+1)*7 + (k_start+1)] = -1
        w3 = [0]*14
        w3[(k_start+1)*2 + 0] = 1
        add_term(u3, v3, w3)
        
        # M4: A11(B10-B00)
        u4 = [0]*6
        u4[1*3 + (j_start+1)] = 1
        v4 = [0]*21
        v4[(j_start+1)*7 + k_start] = 1
        v4[j_start*7 + k_start] = -1
        w4 = [0]*14
        w4[k_start*2 + 0] = 1
        w4[k_start*2 + 1] = 1
        add_term(u4, v4, w4)
        
        # M5: (A00+A01)B11
        u5 = [0]*6
        u5[0*3 + j_start] = 1
        u5[0*3 + (j_start+1)] = 1
        v5 = [0]*21
        v5[(j_start+1)*7 + (k_start+1)] = 1
        w5 = [0]*14
        w5[k_start*2 + 0] = -1
        w5[(k_start+1)*2 + 0] = 1
        add_term(u5, v5, w5)
        
        # M6: (A10-A00)(B00+B10)
        u6 = [0]*6
        u6[1*3 + j_start] = 1
        u6[0*3 + j_start] = -1
        v6 = [0]*21
        v6[j_start*7 + k_start] = 1
        v6[(j_start+1)*7 + k_start] = 1
        w6 = [0]*14
        w6[(k_start+1)*2 + 1] = 1
        add_term(u6, v6, w6)
        
        # M7: (A01-A11)(B01+B11)
        u7 = [0]*6
        u7[0*3 + (j_start+1)] = 1
        u7[1*3 + (j_start+1)] = -1
        v7 = [0]*21
        v7[j_start*7 + (k_start+1)] = 1
        v7[(j_start+1)*7 + (k_start+1)] = 1
        w7 = [0]*14
        w7[k_start*2 + 0] = -1
        add_term(u7, v7, w7)
        
    def add_schoolbook_2x2x1(j_start, k_start):
        # Computes C[0, k_start] += A[0, j_start]B[j_start, k_start] + A[0, j_start+1]B[j_start+1, k_start]
        # C[1, k_start] += A[1, j_start]B[j_start, k_start] + A[1, j_start+1]B[j_start+1, k_start]
        
        # Term 1: A[0, j_start] * B[j_start, k_start]
        u1 = [0]*6
        u1[0*3 + j_start] = 1
        v1 = [0]*21
        v1[j_start*7 + k_start] = 1
        w1 = [0]*14
        w1[k_start*2 + 0] = 1
        add_term(u1, v1, w1)
        
        # Term 2: A[0, j_start+1] * B[j_start+1, k_start]
        u2 = [0]*6
        u2[0*3 + (j_start+1)] = 1
        v2 = [0]*21
        v2[(j_start+1)*7 + k_start] = 1
        w2 = [0]*14
        w2[k_start*2 + 0] = 1
        add_term(u2, v2, w2)
        
        # Term 3: A[1, j_start] * B[j_start, k_start]
        u3 = [0]*6
        u3[1*3 + j_start] = 1
        v3 = [0]*21
        v3[j_start*7 + k_start] = 1
        w3 = [0]*14
        w3[k_start*2 + 1] = 1
        add_term(u3, v3, w3)
        
        # Term 4: A[1, j_start+1] * B[j_start+1, k_start]
        u4 = [0]*6
        u4[1*3 + (j_start+1)] = 1
        v4 = [0]*21
        v4[(j_start+1)*7 + k_start] = 1
        w4 = [0]*14
        w4[k_start*2 + 1] = 1
        add_term(u4, v4, w4)
        
    def add_schoolbook_2x1x7(j_idx, k_start, k_end):
        # Computes C[i, k] += A[i, j_idx] * B[j_idx, k] for i in 0,1 and k in k_start..k_end
        for k in range(k_start, k_end):
            # Term for i=0
            u1 = [0]*6
            u1[0*3 + j_idx] = 1
            v1 = [0]*21
            v1[j_idx*7 + k] = 1
            w1 = [0]*14
            w1[k*2 + 0] = 1
            add_term(u1, v1, w1)
            
            # Term for i=1
            u2 = [0]*6
            u2[1*3 + j_idx] = 1
            v2 = [0]*21
            v2[j_idx*7 + k] = 1
            w2 = [0]*14
            w2[k*2 + 1] = 1
            add_term(u2, v2, w2)

    # 1. A[:, 0:2] * B[0:2, 0:2]
    add_strassen_2x2x2(0, 0)
    
    # 2. A[:, 0:2] * B[0:2, 2:4]
    add_strassen_2x2x2(0, 2)
    
    # 3. A[:, 0:2] * B[0:2, 4:7]
    # Split into 2x2x2 (k=4,5) and 2x2x1 (k=6)
    add_strassen_2x2x2(0, 4)
    add_schoolbook_2x2x1(0, 6)
    
    # 4. A[:, 2] * B[2, 0:7]
    add_schoolbook_2x1x7(2, 0, 7)
    
    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(6)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(21)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(14)]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
