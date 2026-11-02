import numpy as np

def construct():
    n, m, p = 2, 4, 7
    
    # Helper to add a term to U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict mapping index in U (i*m+j) to value
        v_vals: dict mapping index in V (j*p+k) to value
        w_vals: dict mapping index in W (k*n+i) to value
        """
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for idx, val in u_vals.items():
            u_col[idx] = val
        for idx, val in v_vals.items():
            v_col[idx] = val
        for idx, val in w_vals.items():
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Strassen 2x2x2 terms
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in 2x2
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in 2x2
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in 2x2
    
    def strassen_222(offset_a, offset_b, offset_c):
        """
        offset_a: offset for A indices in the full 2x4 matrix (row i, col j)
        offset_b: offset for B indices in the full 4x7 matrix (row j, col k)
        offset_c: offset for C indices in the full 2x7 matrix (row i, col k)
        
        A is 2x2 block. B is 2x2 block. C is 2x2 block.
        """
        # M1 = (A00+A11)(B00+B11)
        add_term(
            {offset_a[0]: 1, offset_a[3]: 1},
            {offset_b[0]: 1, offset_b[3]: 1},
            {offset_c[0]: 1, offset_c[3]: 1} # C00, C11
        )
        # Wait, M1 contributes to C00, C11.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        
        # Let's define the 7 multiplications explicitly as separate terms in the tensor decomposition.
        # The tensor decomposition sums up terms of the form U[:,c] * V[:,c] * W[:,c].
        # Each "multiplication" in Strassen is one such term.
        # The final C is a linear combination of these 7 terms.
        # So we add 7 terms to U, V, W.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11 (with coeff 1)
        # But wait, the W matrix in the tensor decomposition defines the output.
        # The term is sum u_i v_j w_k.
        # If we add a term where W has 1 at C00 and 1 at C11, it adds to both.
        # This is correct for M1.
        
        # Let's re-verify the coefficients.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1 (M1):
        # U: {A00: 1, A11: 1}
        # V: {B00: 1, B11: 1}
        # W: {C00: 1, C11: 1}
        add_term(
            {offset_a[0]: 1, offset_a[3]: 1},
            {offset_b[0]: 1, offset_b[3]: 1},
            {offset_c[0]: 1, offset_c[3]: 1}
        )
        
        # Term 2 (M2): (A10+A11)B00
        # U: {A10: 1, A11: 1}
        # V: {B00: 1}
        # W: {C10: 1, C11: -1}
        add_term(
            {offset_a[2]: 1, offset_a[3]: 1},
            {offset_b[0]: 1},
            {offset_c[2]: 1, offset_c[3]: -1}
        )
        
        # Term 3 (M3): A00(B01-B11)
        # U: {A00: 1}
        # V: {B01: 1, B11: -1}
        # W: {C01: 1, C11: 1}
        add_term(
            {offset_a[0]: 1},
            {offset_b[1]: 1, offset_b[3]: -1},
            {offset_c[1]: 1, offset_c[3]: 1}
        )
        
        # Term 4 (M4): A11(B10-B00)
        # U: {A11: 1}
        # V: {B10: 1, B00: -1}
        # W: {C00: 1, C10: 1}
        add_term(
            {offset_a[3]: 1},
            {offset_b[2]: 1, offset_b[0]: -1},
            {offset_c[0]: 1, offset_c[2]: 1}
        )
        
        # Term 5 (M5): (A00+A01)B11
        # U: {A00: 1, A01: 1}
        # V: {B11: 1}
        # W: {C00: -1, C01: 1}
        add_term(
            {offset_a[0]: 1, offset_a[1]: 1},
            {offset_b[3]: 1},
            {offset_c[0]: -1, offset_c[1]: 1}
        )
        
        # Term 6 (M6): (A10-A00)(B00+B01)
        # U: {A10: 1, A00: -1}
        # V: {B00: 1, B01: 1}
        # W: {C11: 1}
        add_term(
            {offset_a[2]: 1, offset_a[0]: -1},
            {offset_b[0]: 1, offset_b[1]: 1},
            {offset_c[3]: 1}
        )
        
        # Term 7 (M7): (A01-A11)(B10+B11)
        # U: {A01: 1, A11: -1}
        # V: {B10: 1, B11: 1}
        # W: {C00: 1}
        add_term(
            {offset_a[1]: 1, offset_a[3]: -1},
            {offset_b[2]: 1, offset_b[3]: 1},
            {offset_c[0]: 1}
        )

    def matvec_221(offset_a, offset_b, offset_c):
        """
        2x2 * 2x1.
        C0 = A00 B0 + A01 B1
        C1 = A10 B0 + A11 B1
        """
        # Term 1: A00 B0 -> C0
        add_term(
            {offset_a[0]: 1},
            {offset_b[0]: 1},
            {offset_c[0]: 1}
        )
        # Term 2: A01 B1 -> C0
        add_term(
            {offset_a[1]: 1},
            {offset_b[1]: 1},
            {offset_c[0]: 1}
        )
        # Term 3: A10 B0 -> C1
        add_term(
            {offset_a[2]: 1},
            {offset_b[0]: 1},
            {offset_c[1]: 1}
        )
        # Term 4: A11 B1 -> C1
        add_term(
            {offset_a[3]: 1},
            {offset_b[1]: 1},
            {offset_c[1]: 1}
        )

    # Indices for A (2x4)
    # A[i, j] -> index i*4 + j
    # A_L is cols 0,1. A_R is cols 2,3.
    # A_L indices: 0,1 (row 0), 4,5 (row 1)
    # A_R indices: 2,3 (row 0), 6,7 (row 1)
    
    # Indices for B (4x7)
    # B[j, k] -> index j*7 + k
    
    # Indices for C (2x7)
    # C[i, k] -> index k*2 + i (Trace convention W[k*n+i, r])
    
    # Block 1: A * B_1 (cols 0,1 of B)
    # A_L (cols 0,1 of A) * B_1_T (rows 0,1 of B_1)
    # A_L indices: 0,1,4,5
    # B_1_T indices: 0,1,7,8 (rows 0,1, cols 0,1)
    # C_1 (cols 0,1 of C)
    # C_1 indices: 0,1 (col 0), 2,3 (col 1)
    # C_1[i,k] index: k*2 + i
    # C_1[0,0]=0, C_1[0,1]=2
    # C_1[1,0]=1, C_1[1,1]=3
    
    a_L = [0, 1, 4, 5]
    b_1_T = [0, 1, 7, 8]
    c_1 = [0, 2, 1, 3]
    strassen_222(a_L, b_1_T, c_1)
    
    # A_R (cols 2,3 of A) * B_1_B (rows 2,3 of B_1)
    # A_R indices: 2,3,6,7
    # B_1_B indices: 14,15,21,22 (rows 2,3, cols 0,1)
    # C_1 indices same.
    a_R = [2, 3, 6, 7]
    b_1_B = [14, 15, 21, 22]
    strassen_222(a_R, b_1_B, c_1)
    
    # Block 2: A * B_2 (cols 2,3 of B)
    # A_L * B_2_T (rows 0,1, cols 2,3)
    # B_2_T indices: 2,3,9,10
    # C_2 (cols 2,3 of C)
    # C_2[0,2]=4, C_2[0,3]=6
    # C_2[1,2]=5, C_2[1,3]=7
    b_2_T = [2, 3, 9, 10]
    c_2 = [4, 6, 5, 7]
    strassen_222(a_L, b_2_T, c_2)
    
    # A_R * B_2_B (rows 2,3, cols 2,3)
    # B_2_B indices: 16,17,23,24
    b_2_B = [16, 17, 23, 24]
    strassen_222(a_R, b_2_B, c_2)
    
    # Block 3: A * B_3 (cols 4,5,6 of B)
    # Split B_3 into B_3_1 (cols 4,5) and B_3_2 (col 6)
    
    # B_3_1 is 4x2.
    # A_L * B_3_1_T (rows 0,1, cols 4,5)
    # B_3_1_T indices: 4,5,11,12
    # C_3_1 (cols 4,5 of C)
    # C_3_1[0,4]=8, C_3_1[0,5]=10
    # C_3_1[1,4]=9, C_3_1[1,5]=11
    b_3_1_T = [4, 5, 11, 12]
    c_3_1 = [8, 10, 9, 11]
    strassen_222(a_L, b_3_1_T, c_3_1)
    
    # A_R * B_3_1_B (rows 2,3, cols 4,5)
    # B_3_1_B indices: 18,19,25,26
    b_3_1_B = [18, 19, 25, 26]
    strassen_222(a_R, b_3_1_B, c_3_1)
    
    # B_3_2 is 4x1 (col 6).
    # A * B_3_2.
    # A_L * B_3_2_T (rows 0,1, col 6)
    # B_3_2_T indices: 6, 13
    # C_3_2 (col 6 of C)
    # C_3_2[0,6]=12, C_3_2[1,6]=13
    b_3_2_T = [6, 13]
    c_3_2 = [12, 13]
    matvec_221(a_L, b_3_2_T, c_3_2)
    
    # A_R * B_3_2_B (rows 2,3, col 6)
    # B_3_2_B indices: 20, 27
    b_3_2_B = [20, 27]
    matvec_221(a_R, b_3_2_B, c_3_2)
    
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
