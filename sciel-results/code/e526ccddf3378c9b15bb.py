import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    # term index c
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # We have two blocks: Left (cols 0,1 of A, rows 0,1 of B) and Right (cols 2,3 of A, rows 2,3 of B)
    # Block 0: A_L, B_T
    # Block 1: A_R, B_B
    
    # Strassen terms for 2x2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # We will define the coefficients for U, V, W for each of the 7 Strassen terms
    # and 4 schoolbook terms for 2x2x1
    
    # Strassen Coefficients:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Note: A is 2x2, B is 2x2.
    # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    
    # For a given block (offset_a_col, offset_b_row, offset_c_col)
    # offset_a_col: 0 or 2
    # offset_b_row: 0 or 2
    # offset_c_col: 0, 2, 4
    
    def add_strassen_term(col_idx, a_off, b_off, c_off, u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {(i,j): val} for A
        # v_coeffs: dict {(i,j): val} for B
        # w_coeffs: dict {(i,j): val} for C
        
        for (i, j), val in u_coeffs.items():
            U[i * m + (j + a_off), col_idx] = val
        for (i, j), val in v_coeffs.items():
            V[(i + b_off) * p + j, col_idx] = val
        for (i, j), val in w_coeffs.items():
            W[(j + c_off) * n + i, col_idx] = val
            
    # Strassen 1
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    
    # M3 = A00(B01-B11)
    # C01 += M3, C11 += M3
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    
    # M4 = A11(B10-B00)
    # C00 += M4, C10 += M4
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    
    # Schoolbook for 2x2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Term 1: A00 B0 -> C0
    # Term 2: A01 B1 -> C0
    # Term 3: A10 B0 -> C1
    # Term 4: A11 B1 -> C1
    
    sb_u1 = {(0,0): 1}
    sb_v1 = {(0,0): 1}
    sb_w1 = {(0,0): 1}
    
    sb_u2 = {(0,1): 1}
    sb_v2 = {(1,0): 1}
    sb_w2 = {(0,0): 1}
    
    sb_u3 = {(1,0): 1}
    sb_v3 = {(0,0): 1}
    sb_w3 = {(1,0): 1}
    
    sb_u4 = {(1,1): 1}
    sb_v4 = {(1,0): 1}
    sb_w4 = {(1,0): 1}
    
    col = 0
    # Block 0: a_off=0, b_off=0
    # Part 1: 2x2x2 at c_off=0
    add_strassen_term(col, 0, 0, 0, u1, v1, w1); col += 1
    add_strassen_term(col, 0, 0, 0, u2, v2, w2); col += 1
    add_strassen_term(col, 0, 0, 0, u3, v3, w3); col += 1
    add_strassen_term(col, 0, 0, 0, u4, v4, w4); col += 1
    add_strassen_term(col, 0, 0, 0, u5, v5, w5); col += 1
    add_strassen_term(col, 0, 0, 0, u6, v6, w6); col += 1
    add_strassen_term(col, 0, 0, 0, u7, v7, w7); col += 1
    
    # Part 2: 2x2x2 at c_off=2
    add_strassen_term(col, 0, 0, 2, u1, v1, w1); col += 1
    add_strassen_term(col, 0, 0, 2, u2, v2, w2); col += 1
    add_strassen_term(col, 0, 0, 2, u3, v3, w3); col += 1
    add_strassen_term(col, 0, 0, 2, u4, v4, w4); col += 1
    add_strassen_term(col, 0, 0, 2, u5, v5, w5); col += 1
    add_strassen_term(col, 0, 0, 2, u6, v6, w6); col += 1
    add_strassen_term(col, 0, 0, 2, u7, v7, w7); col += 1
    
    # Part 3: 2x2x1 at c_off=4
    add_strassen_term(col, 0, 0, 4, sb_u1, sb_v1, sb_w1); col += 1
    add_strassen_term(col, 0, 0, 4, sb_u2, sb_v2, sb_w2); col += 1
    add_strassen_term(col, 0, 0, 4, sb_u3, sb_v3, sb_w3); col += 1
    add_strassen_term(col, 0, 0, 4, sb_u4, sb_v4, sb_w4); col += 1
    
    # Block 1: a_off=2, b_off=2
    # Part 1: 2x2x2 at c_off=0
    add_strassen_term(col, 2, 2, 0, u1, v1, w1); col += 1
    add_strassen_term(col, 2, 2, 0, u2, v2, w2); col += 1
    add_strassen_term(col, 2, 2, 0, u3, v3, w3); col += 1
    add_strassen_term(col, 2, 2, 0, u4, v4, w4); col += 1
    add_strassen_term(col, 2, 2, 0, u5, v5, w5); col += 1
    add_strassen_term(col, 2, 2, 0, u6, v6, w6); col += 1
    add_strassen_term(col, 2, 2, 0, u7, v7, w7); col += 1
    
    # Part 2: 2x2x2 at c_off=2
    add_strassen_term(col, 2, 2, 2, u1, v1, w1); col += 1
    add_strassen_term(col, 2, 2, 2, u2, v2, w2); col += 1
    add_strassen_term(col, 2, 2, 2, u3, v3, w3); col += 1
    add_strassen_term(col, 2, 2, 2, u4, v4, w4); col += 1
    add_strassen_term(col, 2, 2, 2, u5, v5, w5); col += 1
    add_strassen_term(col, 2, 2, 2, u6, v6, w6); col += 1
    add_strassen_term(col, 2, 2, 2, u7, v7, w7); col += 1
    
    # Part 3: 2x2x1 at c_off=4
    add_strassen_term(col, 2, 2, 4, sb_u1, sb_v1, sb_w1); col += 1
    add_strassen_term(col, 2, 2, 4, sb_u2, sb_v2, sb_w2); col += 1
    add_strassen_term(col, 2, 2, 4, sb_u3, sb_v3, sb_w3); col += 1
    add_strassen_term(col, 2, 2, 4, sb_u4, sb_v4, sb_w4); col += 1
    
    return {"U": U, "V": V, "W": W}
