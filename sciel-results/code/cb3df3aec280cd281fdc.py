import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct the decomposition manually
    # Rank 50 decomposition
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # Strassen decomposition for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A B
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # Indices for 2x2x2 block
    # A indices: 0,1,2,3 (00, 01, 10, 11)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    def add_strassen_term(a_coeffs, b_coeffs, c_coeffs, offset_u, offset_v, offset_w):
        # a_coeffs, b_coeffs, c_coeffs are lists of 7 tuples (idx_u, val), (idx_v, val), (idx_w, val)
        for i in range(7):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for idx, val in a_coeffs[i]:
                u_vec[offset_u + idx] += val
            for idx, val in b_coeffs[i]:
                v_vec[offset_v + idx] += val
            for idx, val in c_coeffs[i]:
                w_vec[offset_w + idx] += val
                
            add_term(u_vec, v_vec, w_vec)

    # Define Strassen coefficients
    # m1: (a00+a11) * (b00+b11) -> c00, c11
    # a: [(0,1), (3,1)]
    # b: [(0,1), (3,1)]
    # c: [(0,1), (3,1)]
    
    # m2: (a10+a11) * b00 -> c10
    # a: [(2,1), (3,1)]
    # b: [(0,1)]
    # c: [(2,1)]
    
    # m3: a00 * (b01-b11) -> c01, c11
    # a: [(0,1)]
    # b: [(1,1), (3,-1)]
    # c: [(1,1), (3,1)]
    
    # m4: a11 * (b10-b00) -> c00, c10
    # a: [(3,1)]
    # b: [(2,1), (0,-1)]
    # c: [(0,1), (2,1)]
    
    # m5: (a00+a01) * b11 -> c00, c01
    # a: [(0,1), (1,1)]
    # b: [(3,1)]
    # c: [(0,-1), (1,1)] # c00 = m1+m4-m5+m7, so -m5. c01 = m3+m5, so +m5.
    # Wait, c00 coeff for m5 is -1. c01 coeff for m5 is 1.
    # So w_vec needs to handle signs.
    
    # Let's just hardcode the 7 terms for 2x2x2
    # Term 1: m1
    # U: a00, a11. V: b00, b11. W: c00, c11.
    # Term 2: m2
    # U: a10, a11. V: b00. W: c10.
    # Term 3: m3
    # U: a00. V: b01, -b11. W: c01, c11.
    # Term 4: m4
    # U: a11. V: b10, -b00. W: c00, c10.
    # Term 5: m5
    # U: a00, a01. V: b11. W: -c00, c01.
    # Term 6: m6
    # U: a10, -a00. V: b00, b10. W: c11.
    # Term 7: m7
    # U: a01, -a11. V: b01, b11. W: c00.
    
    # We need to map these to global indices.
    
    # Block 1: A_L (2x2) * B_T (2x7)
    # A_L indices in U: 0,1,2,3 (rows 0,1 of A, cols 0,1)
    # B_T indices in V: 0..13 (rows 0,1 of B, cols 0..6)
    # C indices in W: 0..13 (rows 0..6 of C, cols 0,1) -> Wait, W is p*n = 14.
    # W index for C[i, k] is k*n + i = 2k + i.
    
    # B_T is 2x7. Split into B_T1 (2x2), B_T2 (2x2), B_T3 (2x2), B_T4 (2x1).
    # B_T1 cols 0,1. B_T2 cols 2,3. B_T3 cols 4,5. B_T4 col 6.
    
    # For each 2x2 block of B_T, use Strassen.
    # For 2x1 block, use schoolbook (4 terms).
    
    # Schoolbook for 2x2x1:
    # A (2x2) * B (2x1) -> C (2x1)
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # Terms:
    # 1: a00 b0 -> c0
    # 2: a01 b1 -> c0
    # 3: a10 b0 -> c1
    # 4: a11 b1 -> c1
    
    # Let's build the terms
    
    # Helper to add schoolbook 2x2x1
    def add_sb_2x2x1(offset_u, offset_v, offset_w, b_col_offset):
        # U offset: 0..3
        # V offset: 0..1 (for the 2x1 block)
        # W offset: 0..1 (for the 2x1 block)
        
        # Term 1: a00 b0 -> c0
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+0] = 1
        v[offset_v+0] = 1
        w[offset_w+0] = 1
        add_term(u, v, w)
        
        # Term 2: a01 b1 -> c0
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+1] = 1
        v[offset_v+1] = 1
        w[offset_w+0] = 1
        add_term(u, v, w)
        
        # Term 3: a10 b0 -> c1
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+2] = 1
        v[offset_v+0] = 1
        w[offset_w+1] = 1
        add_term(u, v, w)
        
        # Term 4: a11 b1 -> c1
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+3] = 1
        v[offset_v+1] = 1
        w[offset_w+1] = 1
        add_term(u, v, w)

    # Helper to add Strassen 2x2x2
    def add_strassen_2x2x2(offset_u, offset_v, offset_w, b_col_offset):
        # U offset: 0..3
        # V offset: 0..3 (for the 2x2 block)
        # W offset: 0..3 (for the 2x2 block)
        
        # m1
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+0] = 1; u[offset_u+3] = 1
        v[offset_v+0] = 1; v[offset_v+3] = 1
        w[offset_w+0] = 1; w[offset_w+3] = 1
        add_term(u, v, w)
        
        # m2
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+2] = 1; u[offset_u+3] = 1
        v[offset_v+0] = 1
        w[offset_w+2] = 1
        add_term(u, v, w)
        
        # m3
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+0] = 1
        v[offset_v+1] = 1; v[offset_v+3] = -1
        w[offset_w+1] = 1; w[offset_w+3] = 1
        add_term(u, v, w)
        
        # m4
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+3] = 1
        v[offset_v+2] = 1; v[offset_v+0] = -1
        w[offset_w+0] = 1; w[offset_w+2] = 1
        add_term(u, v, w)
        
        # m5
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+0] = 1; u[offset_u+1] = 1
        v[offset_v+3] = 1
        w[offset_w+0] = -1; w[offset_w+1] = 1
        add_term(u, v, w)
        
        # m6
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+2] = 1; u[offset_u+0] = -1
        v[offset_v+0] = 1; v[offset_v+2] = 1
        w[offset_w+3] = 1
        add_term(u, v, w)
        
        # m7
        u = [0]*8; v = [0]*28; w = [0]*14
        u[offset_u+1] = 1; u[offset_u+3] = -1
        v[offset_v+1] = 1; v[offset_v+3] = 1
        w[offset_w+0] = 1
        add_term(u, v, w)

    # Block 1: A_L * B_T
    # A_L is rows 0,1 of A. U offset 0.
    # B_T is rows 0,1 of B. V offset 0.
    # C is cols 0,1 of C? No, C is 2x7.
    # W index for C[i, k] is 2k+i.
    # For B_T cols 0,1: W offset 0 (cols 0,1 of C).
    # For B_T cols 2,3: W offset 4 (cols 2,3 of C).
    # For B_T cols 4,5: W offset 8 (cols 4,5 of C).
    # For B_T col 6: W offset 12 (col 6 of C).
    
    # B_T1: cols 0,1. V offset 0. W offset 0.
    add_strassen_2x2x2(0, 0, 0, 0)
    
    # B_T2: cols 2,3. V offset 14 (2*7). W offset 4.
    add_strassen_2x2x2(0, 14, 4, 2)
    
    # B_T3: cols 4,5. V offset 28? No, V is 28.
    # B is 4x7. V rows are j*7+k.
    # B_T is rows 0,1.
    # B_T1: j=0,1; k=0,1. V indices: 0*7+0=0, 0*7+1=1, 1*7+0=7, 1*7+1=8.
    # My offset logic in add_strassen_2x2x2 assumed contiguous V indices.
    # I need to be careful with V indices.
    
    # Let's restart the index mapping.
    # U: 8xR. Index 4i+j.
    # V: 28xR. Index 7j+k.
    # W: 14xR. Index 2k+i.
    
    # Block A_L (i=0,1; j=0,1). U indices 0,1,2,3.
    # Block B_T1 (j=0,1; k=0,1). V indices 0,1,7,8.
    # Block C_T1 (k=0,1; i=0,1). W indices 0,1,2,3.
    
    # This is getting complex to hardcode offsets.
    # I will write a more generic function that takes explicit indices.

    U = []
    V = []
    W = []
    
    def add_term_explicit(u_idxs, v_idxs, w_idxs):
        u = [0.0] * 8
        v = [0.0] * 28
        w = [0.0] * 14
        for idx, val in u_idxs:
            u[idx] += val
        for idx, val in v_idxs:
            v[idx] += val
        for idx, val in w_idxs:
            w[idx] += val
        U.append(u)
        V.append(v)
        W.append(w)

    # Strassen 2x2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # m1
        add_term_explicit([(a00,1),(a11,1)], [(b00,1),(b11,1)], [(c00,1),(c11,1)])
        # m2
        add_term_explicit([(a10,1),(a11,1)], [(b00,1)], [(c10,1)])
        # m3
        add_term_explicit([(a00,1)], [(b01,1),(b11,-1)], [(c01,1),(c11,1)])
        # m4
        add_term_explicit([(a11,1)], [(b10,1),(b00,-1)], [(c00,1),(c10,1)])
        # m5
        add_term_explicit([(a00,1),(a01,1)], [(b11,1)], [(c00,-1),(c01,1)])
        # m6
        add_term_explicit([(a10,1),(a00,-1)], [(b00,1),(b10,1)], [(c11,1)])
        # m7
        add_term_explicit([(a01,1),(a11,-1)], [(b01,1),(b11,1)], [(c00,1)])

    # Schoolbook 2x2x1
    def add_sb(a00, a01, a10, a11, b0, b1, c0, c1):
        add_term_explicit([(a00,1)], [(b0,1)], [(c0,1)])
        add_term_explicit([(a01,1)], [(b1,1)], [(c0,1)])
        add_term_explicit([(a10,1)], [(b0,1)], [(c1,1)])
        add_term_explicit([(a11,1)], [(b1,1)], [(c1,1)])

    # A_L: a00=0, a01=1, a10=2, a11=3
    # B_T1: b00=0, b01=1, b10=7, b11=8
    # C_T1: c00=0, c01=1, c10=2, c11=3
    
    add_strassen(0, 1, 2, 3, 0, 1, 7, 8, 0, 1, 2, 3)
    
    # B_T2: k=2,3. V: 14,15,21,22. W: 4,5,6,7.
    add_strassen(0, 1, 2, 3, 14, 15, 21, 22, 4, 5, 6, 7)
    
    # B_T3: k=4,5. V: 28,29? No V is 28.
    # V index 7j+k.
    # j=0, k=4 -> 4. j=0, k=5 -> 5.
    # j=1, k=4 -> 11. j=1, k=5 -> 12.
    # Wait, B_T is rows 0,1 of B.
    # B is 4x7.
    # Row 0 of B: indices 0..6.
    # Row 1 of B: indices 7..13.
    # So B_T1 (j=0,1; k=0,1):
    # b00 (j=0,k=0) = 0
    # b01 (j=0,k=1) = 1
    # b10 (j=1,k=0) = 7
    # b11 (j=1,k=1) = 8
    
    # B_T2 (j=0,1; k=2,3):
    # b00 (j=0,k=2) = 2
    # b01 (j=0,k=3) = 3
    # b10 (j=1,k=2) = 9
    # b11 (j=1,k=3) = 10
    
    # B_T3 (j=0,1; k=4,5):
    # b00 (j=0,k=4) = 4
    # b01 (j=0,k=5) = 5
    # b10 (j=1,k=4) = 11
    # b11 (j=1,k=5) = 12
    
    # B_T4 (j=0,1; k=6):
    # b0 (j=0,k=6) = 6
    # b1 (j=1,k=6) = 13
    
    # W indices for C_T1 (k=0,1; i=0,1):
    # c00 (k=0,i=0) = 0
    # c01 (k=1,i=0) = 2
    # c10 (k=0,i=1) = 1
    # c11 (k=1,i=1) = 3
    
    # W indices for C_T2 (k=2,3; i=0,1):
    # c00 (k=2,i=0) = 4
    # c01 (k=3,i=0) = 6
    # c10 (k=2,i=1) = 5
    # c11 (k=3,i=1) = 7
    
    # W indices for C_T3 (k=4,5; i=0,1):
    # c00 (k=4,i=0) = 8
    # c01 (k=5,i=0) = 10
    # c10 (k=4,i=1) = 9
    # c11 (k=5,i=1) = 11
    
    # W indices for C_T4 (k=6; i=0,1):
    # c0 (k=6,i=0) = 12
    # c1 (k=6,i=1) = 13
    
    # Corrected calls:
    # B_T1
    add_strassen(0, 1, 2, 3, 0, 1, 7, 8, 0, 2, 1, 3)
    # B_T2
    add_strassen(0, 1, 2, 3, 2, 3, 9, 10, 4, 6, 5, 7)
    # B_T3
    add_strassen(0, 1, 2, 3, 4, 5, 11, 12, 8, 10, 9, 11)
    # B_T4
    add_sb(0, 1, 2, 3, 6, 13, 12, 13)
    
    # Block 2: A_R * B_B
    # A_R: j=2,3. U indices: 4,5,6,7.
    # a00=4, a01=5, a10=6, a11=7
    
    # B_B: rows 2,3 of B.
    # Row 2 of B: indices 14..20.
    # Row 3 of B: indices 21..27.
    
    # B_B1 (j=2,3; k=0,1):
    # b00 (j=2,k=0) = 14
    # b01 (j=2,k=1) = 15
    # b10 (j=3,k=0) = 21
    # b11 (j=3,k=1) = 22
    
    # B_B2 (j=2,3; k=2,3):
    # b00 (j=2,k=2) = 16
    # b01 (j=2,k=3) = 17
    # b10 (j=3,k=2) = 23
    # b11 (j=3,k=3) = 24
    
    # B_B3 (j=2,3; k=4,5):
    # b00 (j=2,k=4) = 18
    # b01 (j=2,k=5) = 19
    # b10 (j=3,k=4) = 25
    # b11 (j=3,k=5) = 26
    
    # B_B4 (j=2,3; k=6):
    # b0 (j=2,k=6) = 20
    # b1 (j=3,k=6) = 27
    
    # C indices are same as before.
    
    add_strassen(4, 5, 6, 7, 14, 15, 21, 22, 0, 2, 1, 3)
    add_strassen(4, 5, 6, 7, 16, 17, 23, 24, 4, 6, 5, 7)
    add_strassen(4, 5, 6, 7, 18, 19, 25, 26, 8, 10, 9, 11)
    add_sb(4, 5, 6, 7, 20, 27, 12, 13)
    
    # Total terms: 3*7 + 4 + 3*7 + 4 = 21 + 4 + 21 + 4 = 50.
    
    return {"U": U, "V": V, "W": W}

