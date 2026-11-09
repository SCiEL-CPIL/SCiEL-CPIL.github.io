"""Matrix multiplication 4x4 by 4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-minimal decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses recursive Strassen's algorithm.
    Splits the 4x8 matrix into two 4x4 matrices.
    Computes 4x4 * 4x4 using Strassen (rank 49).
    Total rank = 49 + 49 = 98.
    
    Decomposition format:
    U: (n*m, rank) = (16, 98), U[i*m+j, r]
    V: (m*p, rank) = (32, 98), V[j*p+k, r]
    W: (p*n, rank) = (32, 98), W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Helper to add a 2x2 Strassen decomposition for a specific X, Y linear combination
    # x_coeffs: 4x4 matrix (flat 16) representing X = sum(x_coeffs[i,j] * A[i,j])
    # y_coeffs: 4x4 matrix (flat 16) representing Y = sum(y_coeffs[i,j] * B[i,j])
    # c_block: (row_block, col_block) which 2x2 block of C this M_k contributes to
    # c_sign: The sign with which M_k contributes to C[c_block]
    # base_r: The starting column index in U, V, W
    
    def add_strassen_2x2(x_coeffs, y_coeffs, c_block, c_sign, base_r):
        # x_coeffs and y_coeffs are lists of 16 floats
        # We need to generate 7 products for Z = X*Y where X, Y are 2x2 matrices
        # defined by the coefficients.
        
        # Extract X entries (2x2) from x_coeffs
        # X[0,0] is coeff of A[0,0] in x_coeffs? No.
        # x_coeffs maps A indices to X indices.
        # Wait, X is a 2x2 matrix.
        # The linear form for X[0,0] is the sum of x_coeffs[i,j] * A[i,j] for the specific A indices that map to X[0,0]?
        # No, in the recursive step, X is formed from A blocks.
        # Example: X = A_00 + A_11.
        # X[0,0] = A[0,0] + A[2,2].
        # So x_coeffs should represent the full 4x4 linear form that produces the 2x2 matrix X?
        # No, it's easier to think:
        # The 2x2 Strassen operates on X[0,0], X[0,1], X[1,0], X[1,1].
        # We need the linear form for each of these 4 entries in terms of A.
        
        # Let's restructure.
        # Instead of passing x_coeffs (16 floats), pass a list of 4 linear forms (each 16 floats) for X[0,0], X[0,1], X[1,0], X[1,1].
        # Same for Y.
        
        pass # Placeholder, will implement below

    # This approach is getting complex to inline. Let's hardcode the 49 terms.
    # Or, better, implement a generic generator.

    # Let's define the 2x2 Strassen products explicitly.
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # We will create a function that takes the linear forms for X[0,0]..X[1,1] and Y[0,0]..Y[1,1]
    # and adds the 7 rank-1 terms to U, V, W.
    
    def add_2x2_strassen(x_forms, y_forms, c_block_row, c_block_col, c_sign, base_r):
        # x_forms: list of 4 lists of 16 floats. x_forms[0] is coeff for X[0,0], etc.
        # y_forms: list of 4 lists of 16 floats.
        # c_block_row, c_block_col: 0 or 1. Indicates which 2x2 block of C this M_k is.
        # c_sign: +1 or -1.
        
        # Define the 7 Strassen steps
        # Each step has:
        # x_lin: linear combination of x_forms[0..3]
        # y_lin: linear combination of y_forms[0..3]
        # z_contrib: which Z entry (00,01,10,11) and with what sign
        
        steps = [
            # P1
            {
                'x': [1, 0, 0, 1], # X00 + X11
                'y': [1, 0, 0, 1], # Y00 + Y11
                'z': {0: 1, 3: 1}  # Z00 += 1, Z11 += 1 (indices 0,1,2,3 for 00,01,10,11)
            },
            # P2
            {
                'x': [0, 0, 1, 1], # X10 + X11
                'y': [1, 0, 0, 0], # Y00
                'z': {2: 1, 3: -1} # Z10 += 1, Z11 -= 1
            },
            # P3
            {
                'x': [1, 0, 0, 0], # X00
                'y': [0, 1, 0, -1], # Y01 - Y11
                'z': {1: 1, 3: 1}  # Z01 += 1, Z11 += 1
            },
            # P4
            {
                'x': [0, 0, 0, 1], # X11
                'y': [0, 0, 1, -1], # Y10 - Y00
                'z': {0: 1, 2: 1}  # Z00 += 1, Z10 += 1
            },
            # P5
            {
                'x': [1, 1, 0, 0], # X00 + X01
                'y': [0, 0, 0, 1], # Y11
                'z': {0: -1, 1: 1} # Z00 -= 1, Z01 += 1
            },
            # P6
            {
                'x': [0, 0, 1, -1], # X10 - X00 ... wait. X10 is index 2, X00 is index 0.
                # Correct: X10 - X00 -> [ -1, 0, 1, 0 ]
                'x': [-1, 0, 1, 0],
                'y': [1, 1, 0, 0], # Y00 + Y01
                'z': {3: 1}        # Z11 += 1
            },
            # P7
            {
                'x': [0, 1, 0, -1], # X01 - X11 -> X01 is index 1, X11 is index 3.
                # Correct: [ 0, 1, 0, -1 ]
                'y': [0, 0, 1, 1], # Y10 + Y11
                'z': {0: 1}        # Z00 += 1
            }
        ]
        
        for i, step in enumerate(steps):
            r = base_r + i
            
            # Compute U coefficients (16 floats)
            u_coeff = [0.0] * 16
            for idx in range(4):
                c = step['x'][idx]
                if c != 0:
                    for j in range(16):
                        u_coeff[j] += c * x_forms[idx][j]
            
            # Compute V coefficients (16 floats)
            v_coeff = [0.0] * 16
            for idx in range(4):
                c = step['y'][idx]
                if c != 0:
                    for j in range(16):
                        v_coeff[j] += c * y_forms[idx][j]
            
            # Compute W coefficients (16 floats for the 4x4 C block)
            # The Z entry index 0->00, 1->01, 2->10, 3->11
            # Z entry (dr, dc) maps to C block (c_block_row, c_block_col) at local (dr, dc)
            # Global C index: (c_block_row*2 + dr)*4 + (c_block_col*2 + dc)
            # W index: k*n + i = (c_block_col*2 + dc)*4 + (c_block_row*2 + dr)
            
            for z_idx, z_sign in step['z'].items():
                dr = z_idx // 2
                dc = z_idx % 2
                
                c_row = c_block_row * 2 + dr
                c_col = c_block_col * 2 + dc
                
                w_idx = c_col * 4 + c_row
                
                W[w_idx][r] += c_sign * z_sign
            
            # Fill U and V
            for i_u in range(16):
                U[i_u][r] = u_coeff[i_u]
            for i_v in range(16):
                V[i_v][r] = v_coeff[i_v]
    
    # Now define the 4x4 Strassen steps
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We need to define x_forms and y_forms for each of the 7 top-level steps.
    # x_forms is a list of 4 lists of 16 floats.
    # x_forms[0] is the linear form for X[0,0].
    
    # Helper to create a linear form for a 2x2 block of A
    # block_row, block_col: 0 or 1
    # entry_dr, entry_dc: 0 or 1
    def get_a_form(block_row, block_col, dr, dc):
        form = [0.0] * 16
        i = block_row * 2 + dr
        j = block_col * 2 + dc
        form[i * 4 + j] = 1.0
        return form
    
    # Helper to create a linear form for a 2x2 block of B
    def get_b_form(block_row, block_col, dr, dc):
        form = [0.0] * 16
        i = block_row * 2 + dr
        j = block_col * 2 + dc
        form[i * 4 + j] = 1.0
        return form

    # Top-level Strassen steps
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each, we need x_forms (4 forms) and y_forms (4 forms).
    
    # M1: X = A00+A11, Y = B00+B11
    # X[0,0] = A00[0,0] + A11[0,0]
    # X[0,1] = A00[0,1] + A11[0,1]
    # X[1,0] = A00[1,0] + A11[1,0]
    # X[1,1] = A00[1,1] + A11[1,1]
    
    def make_x_form(block1, block2, sign1, sign2):
        # block1, block2: (br, bc)
        # Returns list of 4 forms
        forms = []
        for dr in range(2):
            for dc in range(2):
                f = [0.0] * 16
                if sign1 != 0:
                    br, bc = block1
                    i = br * 2 + dr
                    j = bc * 2 + dc
                    f[i * 4 + j] += sign1
                if sign2 != 0:
                    br, bc = block2
                    i = br * 2 + dr
                    j = bc * 2 + dc
                    f[i * 4 + j] += sign2
                forms.append(f)
        return forms

    def make_y_form(block1, block2, sign1, sign2):
        forms = []
        for dr in range(2):
            for dc in range(2):
                f = [0.0] * 16
                if sign1 != 0:
                    br, bc = block1
                    i = br * 2 + dr
                    j = bc * 2 + dc
                    f[i * 4 + j] += sign1
                if sign2 != 0:
                    br, bc = block2
                    i = br * 2 + dr
                    j = bc * 2 + dc
                    f[i * 4 + j] += sign2
                forms.append(f)
        return forms

    # Define the 7 steps
    # c_block: which 2x2 block of C this M_k contributes to?
    # No, M_k is a 2x2 matrix. It contributes to ALL 4 blocks of C?
    # No, in 4x4 Strassen, M_k is a 2x2 matrix of values.
    # The result C is 4x4.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Here M1, M2... are 2x2 matrices.
    # So M1[0,0] contributes to C[0,0].
    # M1[0,1] contributes to C[0,1]? No.
    # The blocks are:
    # C = [[C00, C01], [C10, C11]]
    # Each Cxy is 2x2.
    # M1 is 2x2.
    # M1 contributes to C00?
    # Formula: C00 = M1 + M4 - M5 + M7.
    # This means the 2x2 block C00 is the sum of the 2x2 matrices M1, M4, -M5, M7.
    # So M1[0,0] contributes to C[0,0] (top-left of C00).
    # M1[0,1] contributes to C[0,1] (top-right of C00).
    # M1[1,0] contributes to C[1,0] (bottom-left of C00).
    # M1[1,1] contributes to C[1,1] (bottom-right of C00).
    
    # So for each M_k, we call add_2x2_strassen with c_block_row=0, c_block_col=0?
    # No, add_2x2_strassen adds to a specific 2x2 block of C.
    # But M_k contributes to C00, C01, C10, C11?
    # No, M_k is a single 2x2 matrix.
    # The formula C00 = M1 + ... means the 2x2 block C00 is the sum.
    # So M1 contributes ONLY to the C00 block?
    # Yes, in the standard Strassen derivation for 2x2 blocks, the result is 2x2 blocks.
    # C00, C01, C10, C11 are the 4 blocks.
    # M1 contributes to C00.
    # M2 contributes to C10?
    # Let's check:
    # C10 = M2 + M4.
    # So M2 contributes to C10.
    # C01 = M3 + M5.
    # So M3 contributes to C01.
    # C11 = M1 - M2 + M3 + M6.
    # So M1 contributes to C11 as well?
    # Yes! M1 contributes to BOTH C00 and C11.
    
    # So we need to call add_2x2_strassen multiple times for the same M_k?
    # Or modify add_2x2_strassen to accept a list of (c_block, sign).
    
    # Let's update add_2x2_strassen to accept a list of contributions.
    
    def add_2x2_strassen_multi(x_forms, y_forms, contributions, base_r):
        # contributions: list of (c_block_row, c_block_col, c_sign)
        
        steps = [
            {'x': [1, 0, 0, 1], 'y': [1, 0, 0, 1], 'z': {0: 1, 3: 1}},
            {'x': [0, 0, 1, 1], 'y': [1, 0, 0, 0], 'z': {2: 1, 3: -1}},
            {'x': [1, 0, 0, 0], 'y': [0, 1, 0, -1], 'z': {1: 1, 3: 1}},
            {'x': [0, 0, 0, 1], 'y': [0, 0, 1, -1], 'z': {0: 1, 2: 1}},
            {'x': [1, 1, 0, 0], 'y': [0, 0, 0, 1], 'z': {0: -1, 1: 1}},
            {'x': [-1, 0, 1, 0], 'y': [1, 1, 0, 0], 'z': {3: 1}},
            {'x': [0, 1, 0, -1], 'y': [0, 0, 1, 1], 'z': {0: 1}}
        ]
        
        for i, step in enumerate(steps):
            r = base_r + i
            
            u_coeff = [0.0] * 16
            for idx in range(4):
                c = step['x'][idx]
                if c != 0:
                    for j in range(16):
                        u_coeff[j] += c * x_forms[idx][j]
            
            v_coeff = [0.0] * 16
            for idx in range(4):
                c = step['y'][idx]
                if c != 0:
                    for j in range(16):
                        v_coeff[j] += c * y_forms[idx][j]
            
            for c_block_row, c_block_col, c_sign in contributions:
                for z_idx, z_sign in step['z'].items():
                    dr = z_idx // 2
                    dc = z_idx % 2
                    
                    c_row = c_block_row * 2 + dr
                    c_col = c_block_col * 2 + dc
                    
                    w_idx = c_col * 4 + c_row
                    
                    W[w_idx][r] += c_sign * z_sign
            
            for i_u in range(16):
                U[i_u][r] = u_coeff[i_u]
            for i_v in range(16):
                V[i_v][r] = v_coeff[i_v]

    # Define the 7 top-level steps and their contributions
    
    # M1: X=A00+A11, Y=B00+B11
    # Contributes to C00 (+1) and C11 (+1)
    x1 = make_x_form((0,0), (1,1), 1, 1)
    y1 = make_y_form((0,0), (1,1), 1, 1)
    add_2x2_strassen_multi(x1, y1, [(0,0,1), (1,1,1)], 0)
    
    # M2: X=A10+A11, Y=B00
    # Contributes to C10 (+1) and C11 (-1)
    x2 = make_x_form((1,0), (1,1), 1, 1)
    y2 = make_y_form((0,0), None, 1, 0) # Handle None
    # Fix make_y_form to handle None
    def make_y_form_safe(b1, b2, s1, s2):
        if b2 is None: s2 = 0
        return make_y_form(b1, b2, s1, s2)
    y2 = make_y_form_safe((0,0), None, 1, 0)
    add_2x2_strassen_multi(x2, y2, [(1,0,1), (1,1,-1)], 7)
    
    # M3: X=A00, Y=B01-B11
    # Contributes to C01 (+1) and C11 (+1)
    x3 = make_x_form((0,0), None, 1, 0)
    y3 = make_y_form_safe((0,1), (1,1), 1, -1)
    add_2x2_strassen_multi(x3, y3, [(0,1,1), (1,1,1)], 14)
    
    # M4: X=A11, Y=B10-B00
    # Contributes to C00 (+1) and C10 (+1)
    x4 = make_x_form((1,1), None, 1, 0)
    y4 = make_y_form_safe((1,0), (0,0), 1, -1)
    add_2x2_strassen_multi(x4, y4, [(0,0,1), (1,0,1)], 21)
    
    # M5: X=A00+A01, Y=B11
    # Contributes to C00 (-1) and C01 (+1)
    x5 = make_x_form((0,0), (0,1), 1, 1)
    y5 = make_y_form_safe((1,1), None, 1, 0)
    add_2x2_strassen_multi(x5, y5, [(0,0,-1), (0,1,1)], 28)
    
    # M6: X=A10-A00, Y=B00+B01
    # Contributes to C11 (+1)
    x6 = make_x_form((1,0), (0,0), 1, -1)
    y6 = make_y_form_safe((0,0), (0,1), 1, 1)
    add_2x2_strassen_multi(x6, y6, [(1,1,1)], 35)
    
    # M7: X=A01-A11, Y=B10+B11
    # Contributes to C00 (+1)
    x7 = make_x_form((0,1), (1,1), 1, -1)
    y7 = make_y_form_safe((1,0), (1,1), 1, 1)
    add_2x2_strassen_multi(x7, y7, [(0,0,1)], 42)
    
    # Now do the second half of B (columns 4-7)
    # We need to shift the V indices.
    # B is 4x8.
    # First half: cols 0-3.
    # Second half: cols 4-7.
    # In the flat index j*8+k, the second half has k in 4..7.
    # Our y_forms are 16 floats (4x4).
    # We need to map the 4x4 y_form to the 4x8 V matrix.
    # V index: j*8 + k.
    # For the second half, k_global = k_local + 4.
    
    # We can just copy the U and W parts, and shift V.
    # But U and W are the same?
    # U is for A. A is the same.
    # W is for C. C is 4x8.
    # First half C: cols 0-3.
    # Second half C: cols 4-7.
    # So W indices also need to be shifted.
    
    # Let's just re-run the generation with offsets.
    
    # We need a version of add_2x2_strassen_multi that takes offsets for V and W.
    
    def add_2x2_strassen_multi_offset(x_forms, y_forms, contributions, base_r, v_offset, w_offset):
        steps = [
            {'x': [1, 0, 0, 1], 'y': [1, 0, 0, 1], 'z': {0: 1, 3: 1}},
            {'x': [0, 0, 1, 1], 'y': [1, 0, 0, 0], 'z': {2: 1, 3: -1}},
            {'x': [1, 0, 0, 0], 'y': [0, 1, 0, -1], 'z': {1: 1, 3: 1}},
            {'x': [0, 0, 0, 1], 'y': [0, 0, 1, -1], 'z': {0: 1, 2: 1}},
            {'x': [1, 1, 0, 0], 'y': [0, 0, 0, 1], 'z': {0: -1, 1: 1}},
            {'x': [-1, 0, 1, 0], 'y': [1, 1, 0, 0], 'z': {3: 1}},
            {'x': [0, 1, 0, -1], 'y': [0, 0, 1, 1], 'z': {0: 1}}
        ]
        
        for i, step in enumerate(steps):
            r = base_r + i
            
            u_coeff = [0.0] * 16
            for idx in range(4):
                c = step['x'][idx]
                if c != 0:
                    for j in range(16):
                        u_coeff[j] += c * x_forms[idx][j]
            
            v_coeff = [0.0] * 16
            for idx in range(4):
                c = step['y'][idx]
                if c != 0:
                    for j in range(16):
                        v_coeff[j] += c * y_forms[idx][j]
            
            for c_block_row, c_block_col, c_sign in contributions:
                for z_idx, z_sign in step['z'].items():
                    dr = z_idx // 2
                    dc = z_idx % 2
                    
                    c_row = c_block_row * 2 + dr
                    c_col = c_block_col * 2 + dc
                    
                    # Global W index
                    # w_offset is added to the column index of C?
                    # C is 4x8.
                    # First half: c_col in 0..3.
                    # Second half: c_col in 4..7.
                    # So w_offset should be 4 for the second half.
                    # w_idx = (c_col + w_offset)*4 + c_row ?
                    # No, W index is k*n + i.
                    # k is the column index of C.
                    # i is the row index of C.
                    # So w_idx = (c_col + w_offset)*4 + c_row.
                    
                    w_idx = (c_col + w_offset) * 4 + c_row
                    
                    W[w_idx][r] += c_sign * z_sign
            
            for i_u in range(16):
                U[i_u][r] = u_coeff[i_u]
            
            # V index: j*p + k.
            # j is row index of B.
            # k is col index of B.
            # v_offset is added to k.
            for i_v in range(16):
                # i_v is flat index of 4x4 B block.
                # j = i_v // 4
                # k = i_v % 4
                j = i_v // 4
                k = i_v % 4
                v_idx = j * 8 + (k + v_offset)
                V[v_idx][r] = v_coeff[i_v]

    # Second half
    v_off = 4
    w_off = 4
    
    add_2x2_strassen_multi_offset(x1, y1, [(0,0,1), (1,1,1)], 49, v_off, w_off)
    add_2x2_strassen_multi_offset(x2, y2, [(1,0,1), (1,1,-1)], 56, v_off, w_off)
    add_2x2_strassen_multi_offset(x3, y3, [(0,1,1), (1,1,1)], 63, v_off, w_off)
    add_2x2_strassen_multi_offset(x4, y4, [(0,0,1), (1,0,1)], 70, v_off, w_off)
    add_2x2_strassen_multi_offset(x5, y5, [(0,0,-1), (0,1,1)], 77, v_off, w_off)
    add_2x2_strassen_multi_offset(x6, y6, [(1,1,1)], 84, v_off, w_off)
    add_2x2_strassen_multi_offset(x7, y7, [(0,0,1)], 91, v_off, w_off)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
