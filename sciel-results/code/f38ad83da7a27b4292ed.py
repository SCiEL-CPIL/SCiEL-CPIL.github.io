import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using blocks.
    # Total rank is 114.
    rank = 114
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_offset = 0
    
    # Helper to add a sub-decomposition
    def add_sub(n_sub, m_sub, p_sub, i_off, j_off, k_off, U_sub, V_sub, W_sub):
        nonlocal col_offset
        r = U_sub.shape[1]
        # U_sub has shape (n_sub * m_sub, r)
        # We need to place it in U at rows corresponding to i_off, j_off
        # U index is i * m + j
        # V index is j * p + k
        # W index is k * n + i
        
        # Map local indices to global
        for local_u_row in range(n_sub * m_sub):
            # local_u_row corresponds to local i, j
            # We need to map local (i, j) to global (i_off + i, j_off + j)
            # local_u_row = i_local * m_sub + j_local
            i_local = local_u_row // m_sub
            j_local = local_u_row % m_sub
            global_i = i_off + i_local
            global_j = j_off + j_local
            global_u_row = global_i * m + global_j
            
            U[global_u_row, col_offset : col_offset + r] = U_sub[local_u_row, :]
            
        for local_v_row in range(m_sub * p_sub):
            # local_v_row = j_local * p_sub + k_local
            j_local = local_v_row // p_sub
            k_local = local_v_row % p_sub
            global_j = j_off + j_local
            global_k = k_off + k_local
            global_v_row = global_j * p + global_k
            
            V[global_v_row, col_offset : col_offset + r] = V_sub[local_v_row, :]
            
        for local_w_row in range(p_sub * n_sub):
            # local_w_row = k_local * n_sub + i_local
            k_local = local_w_row // n_sub
            i_local = local_w_row % n_sub
            global_k = k_off + k_local
            global_i = i_off + i_local
            global_w_row = global_k * n + global_i
            
            W[global_w_row, col_offset : col_offset + r] = W_sub[local_w_row, :]
            
        col_offset += r

    # Define Strassen for 2x2x2
    def strassen_222():
        # Rank 7 decomposition
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Strassen factors (standard)
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to express these as linear combinations of U, V, W.
        # U columns correspond to A coefficients.
        # V columns correspond to B coefficients.
        # W columns correspond to C coefficients.
        
        # Let's define the 7 multiplications.
        # Each multiplication t uses one column in U, V, W.
        
        # t1: (A11 + A22)(B11 + B22)
        # U: A11=1, A22=1. V: B11=1, B22=1.
        # W: C11 += t1, C12 += t1, C21 += t1, C22 += t1? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to assign coefficients to U, V, W for each of the 7 terms.
        
        # Term 1:
        U_s[0, 0] = 1; U_s[3, 0] = 1  # A11, A22
        V_s[0, 0] = 1; V_s[3, 0] = 1  # B11, B22
        W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1 # C11, C12, C21, C22? 
        # Wait, W[k*n+i]. n=2.
        # W[0] -> C[0,0] (k=0, i=0)
        # W[1] -> C[0,1] (k=1, i=0)
        # W[2] -> C[1,0] (k=0, i=1)
        # W[3] -> C[1,1] (k=1, i=1)
        
        # Correction:
        # M1 contributes to C11, C12, C21, C22?
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1 (col 0):
        W_s[0, 0] = 1  # C11
        W_s[1, 0] = 0  # C12
        W_s[2, 0] = 0  # C21
        W_s[3, 0] = 1  # C22
        
        # Term 2: (A21 + A22)B11
        U_s[2, 1] = 1; U_s[3, 1] = 1  # A21, A22
        V_s[0, 1] = 1  # B11
        W_s[2, 1] = 1  # C21 (M2 is in C21)
        W_s[3, 1] = -1 # C22 (M2 is in C22 with -1)
        
        # Term 3: A11(B21 - B22)
        U_s[0, 2] = 1  # A11
        V_s[2, 2] = 1; V_s[3, 2] = -1 # B21, B22
        W_s[1, 2] = 1  # C12 (M3 is in C12)
        W_s[3, 2] = 1  # C22 (M3 is in C22)
        
        # Term 4: A22(B12 - B11)
        U_s[3, 3] = 1  # A22
        V_s[1, 3] = 1; V_s[0, 3] = -1 # B12, B11
        W_s[0, 3] = 1  # C11 (M4 is in C11)
        W_s[2, 3] = 1  # C21 (M4 is in C21)
        
        # Term 5: (A11 + A12)B22
        U_s[0, 4] = 1; U_s[1, 4] = 1 # A11, A12
        V_s[3, 4] = 1 # B22
        W_s[0, 4] = -1 # C11 (M5 is in C11 with -1)
        W_s[1, 4] = 1  # C12 (M5 is in C12)
        
        # Term 6: (A21 - A11)(B11 + B12)
        U_s[2, 5] = 1; U_s[0, 5] = -1 # A21, A11
        V_s[0, 5] = 1; V_s[1, 5] = 1 # B11, B12
        W_s[3, 5] = 1  # C22 (M6 is in C22)
        
        # Term 7: (A12 - A22)(B21 + B22)
        U_s[1, 6] = 1; U_s[3, 6] = -1 # A12, A22
        V_s[2, 6] = 1; V_s[3, 6] = 1 # B21, B22
        W_s[0, 6] = 1  # C11 (M7 is in C11)
        
        return U_s, V_s, W_s

    # Define Schoolbook for 2x3x3
    def schoolbook_233():
        n_s, m_s, p_s = 2, 3, 3
        r = n_s * m_s * p_s
        U_s = np.zeros((n_s * m_s, r))
        V_s = np.zeros((m_s * p_s, r))
        W_s = np.zeros((p_s * n_s, r))
        col = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U_s[i * m_s + j, col] = 1.0
                    V_s[j * p_s + k, col] = 1.0
                    W_s[k * n_s + i, col] = 1.0
                    col += 1
        return U_s, V_s, W_s

    # 1. A00 B00 (2x2x2) at i=0, j=0, k=0
    U_s, V_s, W_s = strassen_222()
    add_sub(2, 2, 2, 0, 0, 0, U_s, V_s, W_s)
    
    # 2. A00 B01 (2x2x2) at i=0, j=0, k=2
    U_s, V_s, W_s = strassen_222()
    add_sub(2, 2, 2, 0, 0, 2, U_s, V_s, W_s)
    
    # 3. A00 B02 (2x2x2) at i=0, j=0, k=4
    U_s, V_s, W_s = strassen_222()
    add_sub(2, 2, 2, 0, 0, 4, U_s, V_s, W_s)
    
    # 4. A01 B10 (2x3x3) at i=0, j=2, k=0
    U_s, V_s, W_s = schoolbook_233()
    add_sub(2, 3, 3, 0, 2, 0, U_s, V_s, W_s)
    
    # 5. A01 B11 (2x3x3) at i=0, j=2, k=3
    U_s, V_s, W_s = schoolbook_233()
    add_sub(2, 3, 3, 0, 2, 3, U_s, V_s, W_s)
    
    # 6. A10 B00 (2x2x2) at i=2, j=0, k=0
    U_s, V_s, W_s = strassen_222()
    add_sub(2, 2, 2, 2, 0, 0, U_s, V_s, W_s)
    
    # 7. A10 B01 (2x2x2) at i=2, j=0, k=2
    U_s, V_s, W_s = strassen_222()
    add_sub(2, 2, 2, 2, 0, 2, U_s, V_s, W_s)
    
    # 8. A10 B02 (2x2x2) at i=2, j=0, k=4
    U_s, V_s, W_s = strassen_222()
    add_sub(2, 2, 2, 2, 0, 4, U_s, V_s, W_s)
    
    # 9. A11 B10 (2x3x3) at i=2, j=2, k=0
    U_s, V_s, W_s = schoolbook_233()
    add_sub(2, 3, 3, 2, 2, 0, U_s, V_s, W_s)
    
    # 10. A11 B11 (2x3x3) at i=2, j=2, k=3
    U_s, V_s, W_s = schoolbook_233()
    add_sub(2, 3, 3, 2, 2, 3, U_s, V_s, W_s)
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
