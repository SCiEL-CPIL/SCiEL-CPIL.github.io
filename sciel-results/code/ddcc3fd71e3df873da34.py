import numpy as np

def build_strassen_2x2():
    """
    Returns U, V, W for 2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # W[k*2+i]: C00 is k=0,i=0 -> 0. C11 is k=1,i=1 -> 3.
    
    # Term 2: (A10+A11)B00 -> C10(+1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 is k=0,i=1 -> 0*2+1=1? No. C10 means i=1, k=0. W index k*2+i = 0*2+1=1.
    # Wait, C10 is row 1, col 0.
    # W index is k*n + i. n=2. k=0, i=1. Index 1.
    # My previous W[0,0] for C00 (i=0,k=0) was index 0. Correct.
    # My previous W[3,0] for C11 (i=1,k=1) was index 3. Correct.
    # So for C10 (i=1,k=0), index is 1.
    W[1, 1] = 1
    
    # Term 3: A00(B01-B11) -> C01(+1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1 # C01 is i=0,k=1. Index 1*2+0=2.
    
    # Term 4: A11(B10-B00) -> C00(+1), C10(+1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[1, 3] = 1 # C10
    
    # Term 5: (A00+A01)B11 -> C00(-1), C01(+1)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1  # C01
    
    # Term 6: (A10-A00)(B00+B01) -> C11(+1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # Term 7: (A01-A11)(B10+B11) -> C00(+1), C11(+1)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    W[3, 6] = 1 # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Build 4x4 Strassen decomposition (Rank 49)
    U_2, V_2, W_2 = build_strassen_2x2()
    
    # We need to build 4x4 from 2x2.
    # The 4x4 Strassen algorithm uses 7 multiplications of 2x2 matrices.
    # Let's define the 7 combinations for 4x4.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # Combinations:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will construct U_4, V_4, W_4 of shape (16, 49), (16, 49), (16, 49)
    # Note: For 4x4, U index is i*4+j. V index is j*4+k. W index is k*4+i.
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Helper to add a term
    # term_id: 0..6
    # A_coeffs: dict {block_idx: coeff}
    # B_coeffs: dict {block_idx: coeff}
    # C_coeffs: dict {block_idx: coeff}
    
    def add_4x4_term(term_id, A_coeffs, B_coeffs, C_coeffs):
        # For each 2x2 sub-term t in 0..6
        for t in range(7):
            col = term_id * 7 + t
            
            # U part
            for a_block, a_coeff in A_coeffs.items():
                # a_block is 0,1,2,3 corresponding to A00, A01, A10, A11
                # Block (r,c) of A corresponds to rows r*2..r*2+1, cols c*2..c*2+1
                br, bc = divmod(a_block, 2)
                for li in range(2):
                    for lj in range(2):
                        i_global = br * 2 + li
                        j_global = bc * 2 + lj
                        u_idx = i_global * 4 + j_global
                        u_local_idx = li * 2 + lj
                        U_4[u_idx, col] += a_coeff * U_2[u_local_idx, t]
                        
            # V part
            for b_block, b_coeff in B_coeffs.items():
                br, bc = divmod(b_block, 2)
                for li in range(2):
                    for lj in range(2):
                        j_global = br * 2 + li
                        k_global = bc * 2 + lj
                        v_idx = j_global * 4 + k_global
                        v_local_idx = li * 2 + lj
                        V_4[v_idx, col] += b_coeff * V_2[v_local_idx, t]
                        
            # W part
            for c_block, c_coeff in C_coeffs.items():
                br, bc = divmod(c_block, 2)
                for li in range(2):
                    for lj in range(2):
                        # C block (r,c) corresponds to rows r*2..r*2+1, cols c*2..c*2+1
                        # W index is k*4 + i.
                        # i_global is row, k_global is col.
                        i_global = br * 2 + li
                        k_global = bc * 2 + lj
                        w_idx = k_global * 4 + i_global
                        w_local_idx = lj * 2 + li # local W index k*2+i
                        W_4[w_idx, col] += c_coeff * W_2[w_local_idx, t]

    # Define the 7 terms
    # Blocks: 00=0, 01=1, 10=2, 11=3
    
    # M1
    add_4x4_term(0, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    # M2
    add_4x4_term(1, {2: 1, 3: 1}, {0: 1}, {2: 1})
    # M3
    add_4x4_term(2, {0: 1}, {1: 1, 3: -1}, {1: 1})
    # M4
    add_4x4_term(3, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
    # M5
    add_4x4_term(4, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
    # M6
    add_4x4_term(5, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    # M7
    add_4x4_term(6, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1})
    
    # Now construct 4x4x8 decomposition
    # Rank 98
    U_final = np.zeros((16, 98))
    V_final = np.zeros((32, 98))
    W_final = np.zeros((32, 98))
    
    for t in range(49):
        # First half (b=0)
        U_final[:, t] = U_4[:, t]
        # V_4 index j*4+k' -> V_final index j*8 + (4*0 + k')
        for j in range(4):
            for k_prime in range(4):
                V_final[j*8 + k_prime, t] = V_4[j*4 + k_prime, t]
        # W_4 index k'*4+i -> W_final index (4*0+k')*4 + i
        for k_prime in range(4):
            for i in range(4):
                W_final[(k_prime)*4 + i, t] = W_4[k_prime*4 + i, t]
                
        # Second half (b=1)
        U_final[:, 49+t] = U_4[:, t]
        # V_4 index j*4+k' -> V_final index j*8 + (4*1 + k')
        for j in range(4):
            for k_prime in range(4):
                V_final[j*8 + (4 + k_prime), 49+t] = V_4[j*4 + k_prime, t]
        # W_4 index k'*4+i -> W_final index (4*1+k')*4 + i
        for k_prime in range(4):
            for i in range(4):
                W_final[(4 + k_prime)*4 + i, 49+t] = W_4[k_prime*4 + i, t]
                
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
