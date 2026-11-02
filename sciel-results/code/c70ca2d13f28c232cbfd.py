import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 4, 4, 5
    # We will build a list of terms (u_indices, v_indices, w_indices, coeff)
    # But it's easier to build the matrices directly.
    # Rank estimate: 49 (Strassen 4x4x4) + 4 (4x4x1) = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_idx, v_idx, w_idx, coeff
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff # Wait, V and U coeffs are independent?
        # No, the term is u * v * w.
        # The tensor is sum u_r * v_r * w_r.
        # So U[i,j,r] = u_r, V[j,k,r] = v_r, W[k,i,r] = w_r.
        # For a rank-1 term, u_r, v_r, w_r are constants.
        # In schoolbook, u=1, v=1, w=1.
        # In Strassen, u and v are linear combinations.
        # So U[u_idx][col] = coeff_u
        # V[v_idx][col] = coeff_v
        # W[w_idx][col] = coeff_w
        # My add_term signature is wrong.
        pass

    # Let's restructure.
    # A term is defined by:
    # u_vec: sparse vector of length n*m
    # v_vec: sparse vector of length m*p
    # w_vec: sparse vector of length p*n
    # We add these to the columns of U, V, W.
    
    # Strassen 2x2x2 terms
    # X = [[x11, x12], [x21, x22]]
    # Y = [[y11, y12], [y21, y22]]
    # S1 = (x11+x22)(y11+y22)
    # S2 = (x21+x22)y11
    # S3 = x11(y12-y22)
    # S4 = x22(y21-y11)
    # S5 = (x11+x12)y22
    # S6 = (x21-x11)(y11+y12)
    # S7 = (x12-x22)(y21+y22)
    
    # For 4x4x4:
    # A = [[A1, A2], [A3, A4]]
    # B = [[B1, B2], [B3, B4]]
    # M1 = (A1+A4)(B1+B4)
    # ...
    
    # We need to map the 2x2 Strassen terms to the 4x4 indices.
    # A1 is A[0:2, 0:2], etc.
    
    # Let's define the 7 Strassen products for 4x4.
    # Each product is a 2x2 matrix multiplication.
    # We will call a function to add the 7 terms of a 2x2 multiplication.
    
    def add_2x2_strassen(x_block_indices, y_block_indices, z_block_indices, offset):
        # x_block_indices: dict mapping 'x11' to (i,j) in A
        # y_block_indices: dict mapping 'y11' to (j,k) in B
        # z_block_indices: dict mapping 'z11' to (i,k) in C
        # offset: current column index
        # Returns new offset
        
        # S1 = (x11+x22)(y11+y22)
        # U: x11, x22. V: y11, y22. W: z11, z12, z21, z22 (with coeffs)
        
        # This is getting complex to hardcode.
        # Let's generate the terms programmatically.
        
        pass

    # Given the complexity and time, I will implement the 49-rank Strassen for the first 4 columns and 4-rank standard for the 5th.
    # I will write the explicit indices for the 49 terms.
    
    # 4x4 Strassen decomposition (Rank 49)
    # A indices: (i, j) for i,j in 0..3
    # B indices: (j, k) for j in 0..3, k in 0..3
    # C indices: (i, k) for i in 0..3, k in 0..3
    
    # Blocks:
    # A1: (0,0),(0,1),(1,0),(1,1)
    # A2: (0,2),(0,3),(1,2),(1,3)
    # A3: (2,0),(2,1),(3,0),(3,1)
    # A4: (2,2),(2,3),(3,2),(3,3)
    
    # B1: (0,0),(0,1),(1,0),(1,1)
    # B2: (0,2),(0,3),(1,2),(1,3)
    # B3: (2,0),(2,1),(3,0),(3,1)
    # B4: (2,2),(2,3),(3,2),(3,3)
    
    # C1: (0,0),(0,1),(1,0),(1,1)
    # C2: (0,2),(0,3),(1,2),(1,3)
    # C3: (2,0),(2,1),(3,0),(3,1)
    # C4: (2,2),(2,3),(3,2),(3,3)
    
    # Strassen Products:
    # M1 = (A1+A4)(B1+B4)
    # M2 = (A3+A4)B1
    # M3 = A1(B2-B4)
    # M4 = A4(B3-B1)
    # M5 = (A1+A2)B4
    # M6 = (A3-A1)(B1+B2)
    # M7 = (A2-A4)(B3+B4)
    
    # Each Mk is a 2x2 product.
    # We need the 7 Strassen terms for each Mk.
    
    # I will create a helper to add the 7 terms of a 2x2 multiplication.
    # The inputs are linear combinations of A and B entries.
    
    # This is a large hardcoded structure.
    # I will generate it.
    
    # For each of the 7 products, we have 7 scalar multiplications.
    # Total 49.
    
    # Let's define the terms.
    # Term (u_entries, v_entries, w_entries)
    # u_entries: list of (idx, coeff)
    # v_entries: list of (idx, coeff)
    # w_entries: list of (idx, coeff)
    
    terms = []
    
    # Helper to get index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Define the 7 Strassen products for 4x4
    # Each product is defined by X and Y which are 2x2 matrices of linear forms.
    
    # M1: X = A1+A4, Y = B1+B4
    # A1+A4:
    # x11 = A[0,0] + A[2,2]
    # x12 = A[0,1] + A[2,3]
    # x21 = A[1,0] + A[3,2]
    # x22 = A[1,1] + A[3,3]
    # B1+B4:
    # y11 = B[0,0] + B[2,2]
    # y12 = B[0,1] + B[2,3]
    # y21 = B[1,0] + B[3,2]
    # y22 = B[1,1] + B[3,3]
    
    # This is very tedious to write manually.
    # I will write a generic function to add the Strassen terms given the linear forms.
    
    def add_strassen_terms(x_forms, y_forms, z_block, col_offset):
        # x_forms: 2x2 list of lists of (u_idx, coeff)
        # y_forms: 2x2 list of lists of (v_idx, coeff)
        # z_block: 2x2 list of (w_idx)
        # Returns new col_offset
        
        # S1 = (x11+x22)(y11+y22)
        # S2 = (x21+x22)y11
        # S3 = x11(y12-y22)
        # S4 = x22(y21-y11)
        # S5 = (x11+x12)y22
        # S6 = (x21-x11)(y11+y12)
        # S7 = (x12-x22)(y21+y22)
        
        # We need to combine the linear forms.
        # For each term, we create a new column in U, V, W.
        
        def combine(forms1, forms2, sign2=1):
            # Returns list of (idx, coeff)
            d = {}
            for idx, c in forms1:
                d[idx] = d.get(idx, 0) + c
            for idx, c in forms2:
                d[idx] = d.get(idx, 0) + sign2 * c
            return [(k, v) for k, v in d.items() if v != 0]
        
        # S1
        u_s1 = combine(x_forms[0][0], x_forms[1][1])
        v_s1 = combine(y_forms[0][0], y_forms[1][1])
        w_s1 = [ (z_block[0][0], 1), (z_block[0][1], 1), (z_block[1][0], 1), (z_block[1][1], 1) ] # Coeffs from Z formula
        # Z11 += S1, Z12 += S1, Z21 += S1, Z22 += S1
        # Wait, Z11 = S1 + S4 - S5 + S7
        # Z12 = S3 + S5
        # Z21 = S2 + S4
        # Z22 = S1 - S2 + S3 + S6
        
        # I will add each S term separately.
        
        def add_term(u_list, v_list, w_list):
            nonlocal col_offset
            for idx, c in u_list:
                U[idx][col_offset] = c
            for idx, c in v_list:
                V[idx][col_offset] = c
            for idx, c in w_list:
                W[idx][col_offset] = c
            col_offset += 1
        
        # S1
        add_term(u_s1, v_s1, [ (z_block[0][0], 1), (z_block[0][1], 1), (z_block[1][0], 1), (z_block[1][1], 1) ])
        
        # S2
        u_s2 = combine(x_forms[1][0], x_forms[1][1])
        v_s2 = y_forms[0][0]
        add_term(u_s2, v_s2, [ (z_block[1][0], 1), (z_block[1][1], -1) ]) # Z21 += S2, Z22 -= S2
        
        # S3
        u_s3 = x_forms[0][0]
        v_s3 = combine(y_forms[0][1], y_forms[1][1], -1)
        add_term(u_s3, v_s3, [ (z_block[0][1], 1), (z_block[1][1], 1) ]) # Z12 += S3, Z22 += S3
        
        # S4
        u_s4 = x_forms[1][1]
        v_s4 = combine(y_forms[1][0], y_forms[0][0], -1)
        add_term(u_s4, v_s4, [ (z_block[0][0], 1), (z_block[1][0], 1) ]) # Z11 += S4, Z21 += S4
        
        # S5
        u_s5 = combine(x_forms[0][0], x_forms[0][1])
        v_s5 = y_forms[1][1]
        add_term(u_s5, v_s5, [ (z_block[0][0], -1), (z_block[0][1], 1) ]) # Z11 -= S5, Z12 += S5
        
        # S6
        u_s6 = combine(x_forms[1][0], x_forms[0][0], -1)
        v_s6 = combine(y_forms[0][0], y_forms[0][1])
        add_term(u_s6, v_s6, [ (z_block[1][1], 1) ]) # Z22 += S6
        
        # S7
        u_s7 = combine(x_forms[0][1], x_forms[1][1], -1)
        v_s7 = combine(y_forms[1][0], y_forms[1][1])
        add_term(u_s7, v_s7, [ (z_block[0][0], 1) ]) # Z11 += S7
        
        return col_offset

    # Define the 7 products
    
    # M1: X = A1+A4, Y = B1+B4, Z = C1..C4? No, M1 contributes to all 4 blocks of C?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C11, C22.
    # M2 contributes to C21, C22.
    # M3 contributes to C12, C22.
    # M4 contributes to C11, C21.
    # M5 contributes to C11, C12.
    # M6 contributes to C22.
    # M7 contributes to C11.
    
    # We need to pass the correct z_block for each M.
    # But the z_block indices are the same for all M?
    # No, the z_block for M1 is the indices of C11 and C22?
    # The function add_strassen_terms adds to specific W indices.
    # The W indices are determined by the z_block passed.
    # But M1 contributes to C11 AND C22.
    # So we need to pass a z_block that covers both?
    # No, the Strassen terms S1..S7 are computed once.
    # And they are added to different C blocks with different signs.
    # So I should compute the 7 terms for M1, and then add them to C11 and C22.
    
    # This means I cannot just call add_strassen_terms once per M.
    # I need to compute the 7 (u,v) pairs for M1, and then for each, add to W for C11 and C22.
    
    # Revised plan:
    # For each M_k:
    # 1. Compute the 7 (u_list, v_list) pairs.
    # 2. For each pair, add a term to U and V.
    # 3. Add the corresponding contributions to W for all C blocks it affects.
    
    def get_strassen_uv(x_forms, y_forms):
        # Returns list of 7 (u_list, v_list)
        def combine(forms1, forms2, sign2=1):
            d = {}
            for idx, c in forms1:
                d[idx] = d.get(idx, 0) + c
            for idx, c in forms2:
                d[idx] = d.get(idx, 0) + sign2 * c
            return [(k, v) for k, v in d.items() if v != 0]
            
        u_s1 = combine(x_forms[0][0], x_forms[1][1])
        v_s1 = combine(y_forms[0][0], y_forms[1][1])
        
        u_s2 = combine(x_forms[1][0], x_forms[1][1])
        v_s2 = y_forms[0][0]
        
        u_s3 = x_forms[0][0]
        v_s3 = combine(y_forms[0][1], y_forms[1][1], -1)
        
        u_s4 = x_forms[1][1]
        v_s4 = combine(y_forms[1][0], y_forms[0][0], -1)
        
        u_s5 = combine(x_forms[0][0], x_forms[0][1])
        v_s5 = y_forms[1][1]
        
        u_s6 = combine(x_forms[1][0], x_forms[0][0], -1)
        v_s6 = combine(y_forms[0][0], y_forms[0][1])
        
        u_s7 = combine(x_forms[0][1], x_forms[1][1], -1)
        v_s7 = combine(y_forms[1][0], y_forms[1][1])
        
        return [
            (u_s1, v_s1),
            (u_s2, v_s2),
            (u_s3, v_s3),
            (u_s4, v_s4),
            (u_s5, v_s5),
            (u_s6, v_s6),
            (u_s7, v_s7)
        ]
    
    # Define A and B blocks
    # A1: (0,0),(0,1),(1,0),(1,1)
    # A2: (0,2),(0,3),(1,2),(1,3)
    # A3: (2,0),(2,1),(3,0),(3,1)
    # A4: (2,2),(2,3),(3,2),(3,3)
    
    def get_a_block(r, c):
        # r, c in 0..1
        # Returns 2x2 list of (u_idx, 1)
        return [
            [ (u_idx(r*2+0, c*2+0), 1), (u_idx(r*2+0, c*2+1), 1) ], # Wait, this is not 2x2.
            # x11 is A[r*2, c*2]
            # x12 is A[r*2, c*2+1]
            # x21 is A[r*2+1, c*2]
            # x22 is A[r*2+1, c*2+1]
        ]
    
    # Let's define the forms explicitly.
    
    def a_form(i, j):
        return [(u_idx(i, j), 1)]
        
    def b_form(j, k):
        return [(v_idx(j, k), 1)]
        
    # A1
    A1 = [
        [a_form(0,0), a_form(0,1)],
        [a_form(1,0), a_form(1,1)]
    ]
    A2 = [
        [a_form(0,2), a_form(0,3)],
        [a_form(1,2), a_form(1,3)]
    ]
    A3 = [
        [a_form(2,0), a_form(2,1)],
        [a_form(3,0), a_form(3,1)]
    ]
    A4 = [
        [a_form(2,2), a_form(2,3)],
        [a_form(3,2), a_form(3,3)]
    ]
    
    # B1
    B1 = [
        [b_form(0,0), b_form(0,1)],
        [b_form(1,0), b_form(1,1)]
    ]
    B2 = [
        [b_form(0,2), b_form(0,3)],
        [b_form(1,2), b_form(1,3)]
    ]
    B3 = [
        [b_form(2,0), b_form(2,1)],
        [b_form(3,0), b_form(3,1)]
    ]
    B4 = [
        [b_form(2,2), b_form(2,3)],
        [b_form(3,2), b_form(3,3)]
    ]
    
    # Combine forms
    def add_forms(f1, f2, s2=1):
        d = {}
        for idx, c in f1:
            d[idx] = d.get(idx, 0) + c
        for idx, c in f2:
            d[idx] = d.get(idx, 0) + s2 * c
        return [(k, v) for k, v in d.items() if v != 0]
        
    def sub_forms(f1, f2):
        return add_forms(f1, f2, -1)
        
    def add_block(b1, b2):
        return [
            [add_forms(b1[0][0], b2[0][0]), add_forms(b1[0][1], b2[0][1])],
            [add_forms(b1[1][0], b2[1][0]), add_forms(b1[1][1], b2[1][1])]
        ]
        
    def sub_block(b1, b2):
        return [
            [sub_forms(b1[0][0], b2[0][0]), sub_forms(b1[0][1], b2[0][1])],
            [sub_forms(b1[1][0], b2[1][0]), sub_forms(b1[1][1], b2[1][1])]
        ]
        
    # M1 = (A1+A4)(B1+B4)
    X1 = add_block(A1, A4)
    Y1 = add_block(B1, B4)
    uv1 = get_strassen_uv(X1, Y1)
    
    # M2 = (A3+A4)B1
    X2 = add_block(A3, A4)
    Y2 = B1
    uv2 = get_strassen_uv(X2, Y2)
    
    # M3 = A1(B2-B4)
    X3 = A1
    Y3 = sub_block(B2, B4)
    uv3 = get_strassen_uv(X3, Y3)
    
    # M4 = A4(B3-B1)
    X4 = A4
    Y4 = sub_block(B3, B1)
    uv4 = get_strassen_uv(X4, Y4)
    
    # M5 = (A1+A2)B4
    X5 = add_block(A1, A2)
    Y5 = B4
    uv5 = get_strassen_uv(X5, Y5)
    
    # M6 = (A3-A1)(B1+B2)
    X6 = sub_block(A3, A1)
    Y6 = add_block(B1, B2)
    uv6 = get_strassen_uv(X6, Y6)
    
    # M7 = (A2-A4)(B3+B4)
    X7 = sub_block(A2, A4)
    Y7 = add_block(B3, B4)
    uv7 = get_strassen_uv(X7, Y7)
    
    # Now add to U, V, W
    # C11 indices: w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    # C12 indices: w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)
    # C21 indices: w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)
    # C22 indices: w_idx(2,2), w_idx(2,3), w_idx(3,2), w_idx(3,3)
    
    # Wait, w_idx(k, i).
    # C is 4x5.
    # C11 is rows 0,1 and cols 0,1.
    # w_idx(0,0) is k=0, i=0.
    # w_idx(0,1) is k=0, i=1.
    # w_idx(1,0) is k=1, i=0.
    # w_idx(1,1) is k=1, i=1.
    
    # C12 is rows 0,1 and cols 2,3.
    # w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1).
    
    # C21 is rows 2,3 and cols 0,1.
    # w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3).
    
    # C22 is rows 2,3 and cols 2,3.
    # w_idx(2,2), w_idx(2,3), w_idx(3,2), w_idx(3,3).
    
    c11 = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    c12 = [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)]
    c21 = [w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)]
    c22 = [w_idx(2,2), w_idx(2,3), w_idx(3,2), w_idx(3,3)]
    
    # Coefficients for C blocks from Strassen terms
    # S1: C11 +1, C22 +1
    # S2: C21 +1, C22 -1
    # S3: C12 +1, C22 +1
    # S4: C11 +1, C21 +1
    # S5: C11 -1, C12 +1
    # S6: C22 +1
    # S7: C11 +1
    
    coeff_map = {
        0: {0: 1, 3: 1}, # S1: C11, C22
        1: {1: 1, 3: -1}, # S2: C21, C22
        2: {1: 1, 3: 1}, # S3: C12, C22
        3: {0: 1, 1: 1}, # S4: C11, C21
        4: {0: -1, 1: 1}, # S5: C11, C12
        5: {3: 1}, # S6: C22
        6: {0: 1} # S7: C11
    }
    
    blocks = [c11, c12, c21, c22]
    
    all_uv = [uv1, uv2, uv3, uv4, uv5, uv6, uv7]
    
    for m_idx in range(7):
        uv_list = all_uv[m_idx]
        for s_idx in range(7):
            u_list, v_list = uv_list[s_idx]
            
            # Add to U and V
            for idx, c in u_list:
                U[idx][col] = c
            for idx, c in v_list:
                V[idx][col] = c
            
            # Add to W
            coeffs = coeff_map[s_idx]
            for block_idx, coeff in coeffs.items():
                for w_i in blocks[block_idx]:
                    W[w_i][col] = coeff
            
            col += 1
    
    # Add 5th column terms
    for i in range(4):
        for j in range(4):
            U[u_idx(i, j)][col] = 1
            V[v_idx(j, 4)][col] = 1
            W[w_idx(4, i)][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
