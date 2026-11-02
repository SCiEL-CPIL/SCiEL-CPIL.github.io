"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # col: column index
    # coeff: coefficient (can be -1, 0, 1)
    def add_term(u_idx, v_idx, w_idx, col, coeff):
        if coeff != 0:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff

    # 1. A1 * B1 (M_{2,1,5})
    # A1 is col 0 of A. B1 is row 0 of B.
    # C = A1 * B1
    # A_{i,0} * B_{0,k} -> C_{i,k}
    col = 0
    for i in range(n): # i in 0,1
        for k in range(p): # k in 0..4
            u_idx = i * m + 0 # j=0
            v_idx = 0 * p + k # j=0
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx, col, 1)
            col += 1
    # col is now 10

    # 2. A2 * B2a (M_{2,2,2} Strassen)
    # A2 is cols 1,2 of A. B2a is rows 1,2 of B, cols 0,1.
    # A indices: (0,1)->a11, (0,2)->a12, (1,1)->a21, (1,2)->a22
    # B indices: (1,0)->b11, (1,1)->b12, (2,0)->b21, (2,1)->b22
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    def add_strassen_block(col_offset, b_row_start, b_col_start):
        # b_row_start is 1, b_col_start is 0 or 2
        # a_row_start is 0, a_col_start is 1
        a_r, a_c = 0, 1
        b_r, b_c = b_row_start, b_col_start
        
        # Map local indices to global
        # a11: (0,1), a12: (0,2), a21: (1,1), a22: (1,2)
        # b11: (1,bc), b12: (1,bc+1), b21: (2,bc), b22: (2,bc+1)
        
        # M1 = (a11 + a22)(b11 + b22)
        c = col_offset
        add_term(0*m + 1, 1*p + b_c, (b_c)*n + 0, c, 1) # a11, b11 -> C00
        add_term(0*m + 1, 2*p + b_c, (b_c+1)*n + 0, c, 1) # a11, b22? No.
        # Let's do it systematically.
        # U, V, W updates for each M_s.
        
        # M1
        c = col_offset
        U[0*m+1][c] += 1; V[1*p+b_c][c] += 1; W[(b_c)*n+0][c] += 1 # a11 b11
        U[0*m+1][c] += 1; V[1*p+b_c+1][c] += 1; W[(b_c+1)*n+0][c] += 1 # a11 b22? No, b12 is col b_c+1.
        # Wait, b11 is (1,b_c), b12 is (1,b_c+1), b21 is (2,b_c), b22 is (2,b_c+1).
        # M1 = (a11+a22)(b11+b22)
        # Terms: a11b11, a11b22, a22b11, a22b22
        # a11: (0,1), a22: (1,2)
        # b11: (1,bc), b22: (2,bc+1)
        
        # Reset U,V,W for this col to 0 before adding? 
        # The function add_term adds. I should just add the 4 terms.
        # But I already added some. Let's restart this block logic cleanly.
        
    # Let's rewrite the Strassen addition carefully.
    
    def add_strassen(col_offset, b_row_start, b_col_start):
        # b_row_start is 1, b_col_start is 0 or 2
        # a rows 0,1; a cols 1,2
        # b rows 1,2; b cols b_col_start, b_col_start+1
        
        # Indices
        a11 = (0, 1)
        a12 = (0, 2)
        a21 = (1, 1)
        a22 = (1, 2)
        
        b11 = (1, b_col_start)
        b12 = (1, b_col_start + 1)
        b21 = (2, b_col_start)
        b22 = (2, b_col_start + 1)
        
        c11 = (0, b_col_start)
        c12 = (0, b_col_start + 1)
        c21 = (1, b_col_start)
        c22 = (1, b_col_start + 1)
        
        def get_u(i,j): return i*m + j
        def get_v(j,k): return j*p + k
        def get_w(k,i): return k*n + i
        
        # M1 = (a11 + a22)(b11 + b22)
        c = col_offset
        # a11 * b11 -> c11
        U[get_u(*a11)][c] += 1; V[get_v(*b11)][c] += 1; W[get_w(*c11)][c] += 1
        # a11 * b22 -> c12
        U[get_u(*a11)][c] += 1; V[get_v(*b22)][c] += 1; W[get_w(*c12)][c] += 1
        # a22 * b11 -> c21
        U[get_u(*a22)][c] += 1; V[get_v(*b11)][c] += 1; W[get_w(*c21)][c] += 1
        # a22 * b22 -> c22
        U[get_u(*a22)][c] += 1; V[get_v(*b22)][c] += 1; W[get_w(*c22)][c] += 1
        
        # M2 = (a21 + a22)b11
        c = col_offset + 1
        # a21 * b11 -> c21
        U[get_u(*a21)][c] += 1; V[get_v(*b11)][c] += 1; W[get_w(*c21)][c] += 1
        # a22 * b11 -> c21
        U[get_u(*a22)][c] += 1; V[get_v(*b11)][c] += 1; W[get_w(*c21)][c] += 1
        
        # M3 = a11(b12 - b22)
        c = col_offset + 2
        # a11 * b12 -> c12
        U[get_u(*a11)][c] += 1; V[get_v(*b12)][c] += 1; W[get_w(*c12)][c] += 1
        # a11 * b22 -> c12 (with -1)
        U[get_u(*a11)][c] += 1; V[get_v(*b22)][c] += 1; W[get_w(*c12)][c] -= 1
        
        # M4 = a22(b21 - b11)
        c = col_offset + 3
        # a22 * b21 -> c22
        U[get_u(*a22)][c] += 1; V[get_v(*b21)][c] += 1; W[get_w(*c22)][c] += 1
        # a22 * b11 -> c22 (with -1)
        U[get_u(*a22)][c] += 1; V[get_v(*b11)][c] += 1; W[get_w(*c22)][c] -= 1
        
        # M5 = (a11 + a12)b22
        c = col_offset + 4
        # a11 * b22 -> c12
        U[get_u(*a11)][c] += 1; V[get_v(*b22)][c] += 1; W[get_w(*c12)][c] += 1
        # a12 * b22 -> c12
        U[get_u(*a12)][c] += 1; V[get_v(*b22)][c] += 1; W[get_w(*c12)][c] += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        c = col_offset + 5
        # a21 * b11 -> c21
        U[get_u(*a21)][c] += 1; V[get_v(*b11)][c] += 1; W[get_w(*c21)][c] += 1
        # a21 * b12 -> c22? No.
        # C22 = M1 - M2 + M3 + M6.
        # C21 = M2 + M4.
        # Let's check Strassen formulas again.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M6 contributes to C22.
        # M6 = (a21 - a11)(b11 + b12)
        # Terms: a21b11, a21b12, -a11b11, -a11b12
        # a21b11 -> C22?
        # In C22 formula, M6 is added.
        # So a21b11 should contribute to C22?
        # Wait.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's trace M6 to C22.
        # M6 = (A21-A11)(B11+B12)
        # This product is added to C22.
        # So the terms of M6 should map to C22.
        # U, V, W for M6 should have W index for C22.
        
        # Correction:
        # For M6:
        # a21 * b11 -> C22
        # a21 * b12 -> C22
        # -a11 * b11 -> C22
        # -a11 * b12 -> C22
        
        # Let's re-verify M3 to C12 and C22.
        # C12 = M3 + M5.
        # C22 = ... + M3 + ...
        # So M3 contributes to both C12 and C22?
        # No, M3 is a scalar.
        # C12 gets +M3.
        # C22 gets +M3.
        # So W for M3 should have +1 at C12 and +1 at C22?
        # No, W is a vector.
        # W[index] is the coefficient for that output.
        # If M3 contributes to C12 and C22, then W[C12] += 1 and W[C22] += 1.
        
        # Let's re-do the W assignments.
        
        # M1: C11(+1), C22(-1)
        # M2: C21(+1), C22(-1)
        # M3: C12(+1), C22(+1)
        # M4: C11(+1), C21(+1)
        # M5: C11(-1), C12(+1)
        # M6: C22(+1)
        # M7: C11(+1)
        
        # Let's re-implement the Strassen block with these W coefficients.
        
    # I will rewrite the whole construct function to be clean.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. A1 B1
    col = 0
    for i in range(n):
        for k in range(p):
            U[i*m + 0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # 2. Strassen for A2 B2a (b_col_start=0)
    def add_strassen_block(col_offset, b_col_start):
        # a rows 0,1; a cols 1,2
        # b rows 1,2; b cols bc, bc+1
        bc = b_col_start
        
        def get_u(i,j): return i*m + j
        def get_v(j,k): return j*p + k
        def get_w(k,i): return k*n + i
        
        # M1
        c = col_offset
        U[get_u(0,1)][c] += 1; V[get_v(1,bc)][c] += 1; W[get_w(bc,0)][c] += 1
        U[get_u(0,1)][c] += 1; V[get_v(1,bc+1)][c] += 1; W[get_w(bc+1,0)][c] += 1
        U[get_u(1,2)][c] += 1; V[get_v(1,bc)][c] += 1; W[get_w(bc,1)][c] += 1
        U[get_u(1,2)][c] += 1; V[get_v(1,bc+1)][c] += 1; W[get_w(bc+1,1)][c] += 1
        # W coeffs for M1: C11(+1), C22(-1)
        # I added +1 to C11 and C22? No, I added to specific indices.
        # C11 is (0,bc). C22 is (1,bc+1).
        # I need to set W[get_w(bc,0)] += 1 and W[get_w(bc+1,1)] -= 1.
        # But I also have terms for C12 and C21?
        # M1 = (a11+a22)(b11+b22)
        # The product is a scalar.
        # It contributes to C11 and C22.
        # So W[get_w(bc,0)] should be +1.
        # W[get_w(bc+1,1)] should be -1.
        # The U and V parts define the scalar.
        # The W part distributes it.
        
        # My previous U,V additions were for the scalar calculation.
        # But I added multiple U,V terms to the same column.
        # This is correct for the scalar value.
        # But W must be set correctly.
        
        # Let's clear and redo W for this column.
        # Actually, I should just add to W.
        # But I already added 1 to C11 and C22?
        # No, I added to C11, C12, C21, C22 in the U,V loop?
        # No, U,V loop just sets the scalar.
        # W must be set to the distribution coefficients.
        
        # Let's restart the Strassen addition logic to be precise.
        
    # I will implement the function carefully.
    
    def add_strassen_block_v2(col_offset, b_col_start):
        bc = b_col_start
        def get_u(i,j): return i*m + j
        def get_v(j,k): return j*p + k
        def get_w(k,i): return k*n + i
        
        # M1: (a11+a22)(b11+b22) -> C11(+1), C22(-1)
        c = col_offset
        U[get_u(0,1)][c] += 1; V[get_v(1,bc)][c] += 1
        U[get_u(0,1)][c] += 1; V[get_v(1,bc+1)][c] += 1
        U[get_u(1,2)][c] += 1; V[get_v(1,bc)][c] += 1
        U[get_u(1,2)][c] += 1; V[get_v(1,bc+1)][c] += 1
        W[get_w(bc,0)][c] += 1
        W[get_w(bc+1,1)][c] -= 1
        
        # M2: (a21+a22)b11 -> C21(+1), C22(-1)
        c = col_offset + 1
        U[get_u(1,1)][c] += 1; V[get_v(1,bc)][c] += 1
        U[get_u(1,2)][c] += 1; V[get_v(1,bc)][c] += 1
        W[get_w(bc,1)][c] += 1
        W[get_w(bc+1,1)][c] -= 1
        
        # M3: a11(b12-b22) -> C12(+1), C22(+1)
        c = col_offset + 2
        U[get_u(0,1)][c] += 1; V[get_v(1,bc+1)][c] += 1
        U[get_u(0,1)][c] += 1; V[get_v(2,bc+1)][c] += 1
        W[get_w(bc+1,0)][c] += 1
        W[get_w(bc+1,1)][c] += 1
        
        # M4: a22(b21-b11) -> C11(+1), C21(+1)
        c = col_offset + 3
        U[get_u(1,2)][c] += 1; V[get_v(2,bc)][c] += 1
        U[get_u(1,2)][c] += 1; V[get_v(1,bc)][c] += 1
        W[get_w(bc,0)][c] += 1
        W[get_w(bc,1)][c] += 1
        
        # M5: (a11+a12)b22 -> C11(-1), C12(+1)
        c = col_offset + 4
        U[get_u(0,1)][c] += 1; V[get_v(2,bc+1)][c] += 1
        U[get_u(0,2)][c] += 1; V[get_v(2,bc+1)][c] += 1
        W[get_w(bc,0)][c] -= 1
        W[get_w(bc+1,0)][c] += 1
        
        # M6: (a21-a11)(b11+b12) -> C22(+1)
        c = col_offset + 5
        U[get_u(1,1)][c] += 1; V[get_v(1,bc)][c] += 1
        U[get_u(1,1)][c] += 1; V[get_v(1,bc+1)][c] += 1
        U[get_u(0,1)][c] -= 1; V[get_v(1,bc)][c] += 1
        U[get_u(0,1)][c] -= 1; V[get_v(1,bc+1)][c] += 1
        W[get_w(bc+1,1)][c] += 1
        
        # M7: (a12-a22)(b21+b22) -> C11(+1)
        c = col_offset + 6
        U[get_u(0,2)][c] += 1; V[get_v(2,bc)][c] += 1
        U[get_u(0,2)][c] += 1; V[get_v(2,bc+1)][c] += 1
        U[get_u(1,2)][c] -= 1; V[get_v(2,bc)][c] += 1
        U[get_u(1,2)][c] -= 1; V[get_v(2,bc+1)][c] += 1
        W[get_w(bc,0)][c] += 1

    add_strassen_block_v2(10, 0)
    add_strassen_block_v2(17, 2)
    
    # 3. Schoolbook for A2 B2c (b_col_start=4, only 1 col)
    # A2 cols 1,2. B2c row 1,2 col 4.
    # C2c col 4.
    # C04 = A01 B14 + A02 B24
    # C14 = A11 B14 + A12 B24
    
    c = 24
    U[0*m+1][c] = 1; V[1*p+4][c] = 1; W[4*n+0][c] = 1
    c = 25
    U[0*m+2][c] = 1; V[2*p+4][c] = 1; W[4*n+0][c] = 1
    c = 26
    U[1*m+1][c] = 1; V[1*p+4][c] = 1; W[4*n+1][c] = 1
    c = 27
    U[1*m+2][c] = 1; V[2*p+4][c] = 1; W[4*n+1][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
