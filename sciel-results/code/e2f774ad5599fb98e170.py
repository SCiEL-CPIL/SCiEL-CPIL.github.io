def generate_strassen_4x4():
    # Returns U, V, W for 4x4 * 4x4 = 4x4 with rank 49
    # U: 16 x 49
    # V: 16 x 49
    # W: 16 x 49
    
    # 1. Define inner Strassen 2x2 decomposition
    # Inputs A00, A01, A10, A11
    # Inputs B00, B01, B10, B11
    # 7 multiplications M1..M7
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We represent each M as a pair of linear forms (u, v)
    # u is a vector of coeffs for A, v is a vector of coeffs for B
    
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices: 0:B00, 1:B01, 2:B10, 3:B11
    
    M_forms = []
    
    # M1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    M_forms.append((u1, v1))
    
    # M2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    M_forms.append((u2, v2))
    
    # M3
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    M_forms.append((u3, v3))
    
    # M4
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    M_forms.append((u4, v4))
    
    # M5
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    M_forms.append((u5, v5))
    
    # M6
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    M_forms.append((u6, v6))
    
    # M7
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    M_forms.append((u7, v7))
    
    # Output coeffs for each M
    # C00: M1(1), M2(0), M3(0), M4(1), M5(-1), M6(0), M7(1)
    # C01: M1(0), M2(0), M3(1), M4(0), M5(1), M6(0), M7(0)
    # C10: M1(0), M2(1), M3(0), M4(1), M5(0), M6(0), M7(0)
    # C11: M1(1), M2(-1), M3(1), M4(0), M5(0), M6(1), M7(0)
    
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Now apply this to 4x4 blocks
    # Outer blocks A00, A01, A10, A11 are 2x2 matrices.
    # Let's denote the 16 elements of the 4x4 A as A[0..15]
    # A[I*2+i'][J*2+j']
    # We will build the 49 terms.
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Outer Strassen steps
    # We need to define the outer linear combinations of blocks.
    # Let Bk be the 2x2 block.
    # Outer M1: (A00+A11)(B00+B11)
    # Outer M2: (A10+A11)B00
    # Outer M3: A00(B01-B11)
    # Outer M4: A11(B10-B00)
    # Outer M5: (A00+A01)B11
    # Outer M6: (A10-A00)(B00+B01)
    # Outer M7: (A01-A11)(B10+B11)
    
    # For each outer step k, we have a "U_block" (2x2 linear combo of A blocks)
    # and "V_block" (2x2 linear combo of B blocks).
    # We then apply the inner Strassen decomposition to the product U_block * V_block.
    # The result is a 2x2 matrix P_k.
    # P_k is added to the final C according to the outer C_coeffs.
    
    # Let's define the outer block coefficients.
    # A block indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B block indices: 0:B00, 1:B01, 2:B10, 3:B11
    
    outer_u_blocks = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    outer_v_blocks = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    # For each outer step k (0..6):
    for k in range(7):
        # Construct the 2x2 matrices U_k and V_k (as linear forms on the 16 elements)
        # U_k is a 2x2 matrix where each entry is a linear form on A's 16 elements.
        # V_k is a 2x2 matrix where each entry is a linear form on B's 16 elements.
        
        # U_k[I][J] is the linear form for the block at position I,J of the "U" matrix in step k.
        # Wait, the outer Strassen formula operates on blocks.
        # The "U" matrix in step k is a 2x2 matrix of blocks.
        # Let's call the blocks of A: A_BL[0..3] corresponding to A00, A01, A10, A11.
        # U_k is a 2x2 matrix of linear combinations of these blocks.
        # U_k[0][0] = sum(outer_u_blocks[k][b] * A_BL[b])
        # But A_BL[b] is itself a 2x2 matrix of scalars.
        # So U_k[I][J] is a 2x2 matrix of linear forms on the 16 scalars of A.
        
        # Actually, it's simpler to say:
        # The product P_k = U_k * V_k is a 2x2 matrix of blocks.
        # Each block (I,J) of P_k is the product of two 2x2 matrices of linear forms.
        # We decompose this block product using inner Strassen.
        
        # Let's iterate over the 7 inner Strassen steps s (0..6) for this outer step k.
        # The inner Strassen step s computes a scalar multiplication term.
        # This term contributes to the blocks of P_k.
        
        # For a fixed k, we have 7 inner terms.
        # Total 49 terms.
        
        for s in range(7):
            term_idx = k * 7 + s
            
            # We need to find the coefficients for U_term and V_term.
            # The inner Strassen step s involves linear forms u_s and v_s on the 2x2 blocks of U_k and V_k.
            # u_s is a 2x2 matrix of coefficients for the blocks of U_k.
            # v_s is a 2x2 matrix of coefficients for the blocks of V_k.
            
            # From M_forms[s]:
            # u_s is the 4-vector of coeffs for the 4 blocks of the first matrix.
            # v_s is the 4-vector of coeffs for the 4 blocks of the second matrix.
            
            u_s_vec = M_forms[s][0]
            v_s_vec = M_forms[s][1]
            
            # u_s_vec[0] is coeff for U_k[0][0]
            # u_s_vec[1] is coeff for U_k[0][1]
            # u_s_vec[2] is coeff for U_k[1][0]
            # u_s_vec[3] is coeff for U_k[1][1]
            
            # U_k[I][J] is a linear combination of A blocks.
            # U_k[I][J] = sum_{b in 0..3} (outer_u_blocks[k] is not a matrix, it's a vector of 4 coeffs for the 4 blocks of the "U" matrix in the outer step?)
            
            # Correction: The outer Strassen formula defines 7 products of 2x2 block matrices.
            # For step k, the "U" matrix is a 2x2 matrix of blocks.
            # Let's define the 2x2 matrix of block coefficients for U in step k.
            # outer_u_blocks[k] is a vector of 4 coefficients? No.
            # In standard Strassen, M1 = (A11+A22)(B11+B22).
            # This is a product of two 2x2 matrices? No, A11+A22 is a scalar if A is 2x2.
            # But here A is 4x4, so A11 is a 2x2 block.
            # So A11+A22 is a 2x2 matrix.
            # So the "U" matrix in step k is a 2x2 matrix of blocks.
            # Let's denote the blocks of A as A_00, A_01, A_10, A_11.
            # For M1: U_1 = A_00 + A_11. This is a 2x2 matrix.
            # So U_1 is a single 2x2 matrix, not a 2x2 matrix of blocks.
            
            # Ah, I confused myself.
            # In the 4x4 case, we view A as 2x2 blocks of 2x2 matrices.
            # Let A = [[A_00, A_01], [A_10, A_11]].
            # Strassen M1 = (A_00 + A_11) (B_00 + B_11).
            # Here A_00 + A_11 is a 2x2 matrix (sum of two 2x2 blocks).
            # So the "U" factor in M1 is the 2x2 matrix (A_00 + A_11).
            # The "V" factor is (B_00 + B_11).
            # The product is a 2x2 matrix.
            
            # So for each outer step k, we have a 2x2 matrix U_k and a 2x2 matrix V_k.
            # We need to compute U_k * V_k.
            # We use inner Strassen to compute this 2x2 matrix product.
            # The inner Strassen algorithm takes two 2x2 matrices and produces 7 scalar multiplications.
            
            # So, for outer step k:
            # U_k is a 2x2 matrix of linear forms on A's 16 elements.
            # V_k is a 2x2 matrix of linear forms on B's 16 elements.
            
            # We need to define U_k and V_k explicitly.
            # U_k[I][J] is a linear form.
            
            # For M1: U_1 = A_00 + A_11.
            # U_1[0][0] = A[0,0] + A[2,2]
            # U_1[0][1] = A[0,1] + A[2,3]
            # U_1[1][0] = A[1,0] + A[3,2]
            # U_1[1][1] = A[1,1] + A[3,3]
            
            # Let's define the 2x2 block coefficients for each outer step.
            # Let outer_u_mat[k] be a 2x2 matrix of 4-vectors (coeffs for A_00, A_01, A_10, A_11).
            # No, simpler: outer_u_mat[k] is a 2x2 matrix of 16-vectors? No.
            
            # Let's just compute the 16-vector for each entry of U_k.
            
            # Helper to get 16-vector for a block A_ij
            def block_vec(i, j):
                v = [0]*16
                v[i*4+j] = 1
                return v
            
            # A_00: i=0,j=0. A_01: i=0,j=1. A_10: i=1,j=0. A_11: i=1,j=1.
            # Note: Block indices I,J correspond to block rows/cols.
            # Element (I,J) of the block matrix is a 2x2 matrix.
            # A_00 is the top-left 2x2. Elements 0,1,4,5.
            # A_01 is top-right 2x2. Elements 2,3,6,7.
            # A_10 is bottom-left 2x2. Elements 8,9,12,13.
            # A_11 is bottom-right 2x2. Elements 10,11,14,15.
            
            # Let's define a function to add a scaled block to a vector.
            def add_block(vec, block_idx, scale):
                # block_idx 0: A00, 1: A01, 2: A10, 3: A11
                # Indices in 16-vector:
                # A00: 0,1,4,5
                # A01: 2,3,6,7
                # A10: 8,9,12,13
                # A11: 10,11,14,15
                ranges = [
                    [0,1,4,5],
                    [2,3,6,7],
                    [8,9,12,13],
                    [10,11,14,15]
                ]
                for idx in ranges[block_idx]:
                    vec[idx] += scale
                return vec
            
            # Now define U_k and V_k for each k.
            # U_k is a 2x2 matrix of 16-vectors.
            
            U_mats = []
            V_mats = []
            
            for k in range(7):
                U_k = [[ [0]*16 for _ in range(2) ] for _ in range(2)]
                V_k = [[ [0]*16 for _ in range(2) ] for _ in range(2)]
                
                # M1: (A00+A11)(B00+B11)
                if k == 0:
                    # U_1 = A00+A11
                    for I in range(2):
                        for J in range(2):
                            add_block(U_k[I][J], 0, 1)
                            add_block(U_k[I][J], 3, 1)
                    # V_1 = B00+B11
                    for I in range(2):
                        for J in range(2):
                            add_block(V_k[I][J], 0, 1)
                            add_block(V_k[I][J], 3, 1)
                # M2: (A10+A11)B00
                elif k == 1:
                    for I in range(2):
                        for J in range(2):
                            add_block(U_k[I][J], 2, 1)
                            add_block(U_k[I][J], 3, 1)
                    for I in range(2):
                        for J in range(2):
                            add_block(V_k[I][J], 0, 1)
                # M3: A00(B01-B11)
                elif k == 2:
                    for I in range(2):
                        for J in range(2):
                            add_block(U_k[I][J], 0, 1)
                    for I in range(2):
                        for J in range(2):
                            add_block(V_k[I][J], 1, 1)
                            add_block(V_k[I][J], 3, -1)
                # M4: A11(B10-B00)
                elif k == 3:
                    for I in range(2):
                        for J in range(2):
                            add_block(U_k[I][J], 3, 1)
                    for I in range(2):
                        for J in range(2):
                            add_block(V_k[I][J], 2, 1)
                            add_block(V_k[I][J], 0, -1)
                # M5: (A00+A01)B11
                elif k == 4:
                    for I in range(2):
                        for J in range(2):
                            add_block(U_k[I][J], 0, 1)
                            add_block(U_k[I][J], 1, 1)
                    for I in range(2):
                        for J in range(2):
                            add_block(V_k[I][J], 3, 1)
                # M6: (A10-A00)(B00+B01)
                elif k == 5:
                    for I in range(2):
                        for J in range(2):
                            add_block(U_k[I][J], 2, 1)
                            add_block(U_k[I][J], 0, -1)
                    for I in range(2):
                        for J in range(2):
                            add_block(V_k[I][J], 0, 1)
                            add_block(V_k[I][J], 1, 1)
                # M7: (A01-A11)(B10+B11)
                elif k == 6:
                    for I in range(2):
                        for J in range(2):
                            add_block(U_k[I][J], 1, 1)
                            add_block(U_k[I][J], 3, -1)
                    for I in range(2):
                        for J in range(2):
                            add_block(V_k[I][J], 2, 1)
                            add_block(V_k[I][J], 3, 1)
                
                U_mats.append(U_k)
                V_mats.append(V_k)
            
            # Now, for each k, we apply inner Strassen to U_k * V_k.
            # Inner Strassen steps s=0..6.
            # M_forms[s] gives u_s_vec and v_s_vec (4-vectors).
            # u_s_vec[0] is coeff for U_k[0][0]
            # u_s_vec[1] is coeff for U_k[0][1]
            # u_s_vec[2] is coeff for U_k[1][0]
            # u_s_vec[3] is coeff for U_k[1][1]
            
            for s in range(7):
                term_idx = k * 7 + s
                u_s_vec = M_forms[s][0]
                v_s_vec = M_forms[s][1]
                
                # Construct U_term (16-vector)
                U_term = [0]*16
                # u_s_vec[0] * U_k[0][0]
                for i in range(16):
                    U_term[i] += u_s_vec[0] * U_mats[k][0][0][i]
                # u_s_vec[1] * U_k[0][1]
                for i in range(16):
                    U_term[i] += u_s_vec[1] * U_mats[k][0][1][i]
                # u_s_vec[2] * U_k[1][0]
                for i in range(16):
                    U_term[i] += u_s_vec[2] * U_mats[k][1][0][i]
                # u_s_vec[3] * U_k[1][1]
                for i in range(16):
                    U_term[i] += u_s_vec[3] * U_mats[k][1][1][i]
                
                # Construct V_term (16-vector)
                V_term = [0]*16
                for i in range(16):
                    V_term[i] += v_s_vec[0] * V_mats[k][0][0][i]
                for i in range(16):
                    V_term[i] += v_s_vec[1] * V_mats[k][0][1][i]
                for i in range(16):
                    V_term[i] += v_s_vec[2] * V_mats[k][1][0][i]
                for i in range(16):
                    V_term[i] += v_s_vec[3] * V_mats[k][1][1][i]
                
                # Set U and V for this term
                for i in range(16):
                    U[i][term_idx] = U_term[i]
                    V[i][term_idx] = V_term[i]
                
                # Now compute W_term.
                # The product U_term * V_term contributes to the 2x2 matrix P_k.
                # P_k is a 2x2 matrix of blocks.
                # The inner Strassen output coeffs C_coeffs[s] tell us how the 7 inner terms contribute to the 4 blocks of P_k.
                # C_coeffs is a 4x7 matrix. C_coeffs[c_block][s] is the coeff of term s for block c_block.
                # c_block 0: P_k[0][0], 1: P_k[0][1], 2: P_k[1][0], 3: P_k[1][1].
                
                # P_k is then added to the final C according to outer C_coeffs.
                # outer C_coeffs is 4x7. outer_C_coeffs[c_block][k] is the coeff of outer term k for block c_block.
                
                # So for each of the 4 blocks of the final C (C_00, C_01, C_10, C_11):
                # The contribution from term (k,s) is:
                # Sum over inner block b_in (0..3) of:
                #   C_coeffs[b_in][s] * outer_C_coeffs[b_in][k] * (Block b_in of P_k)
                
                # Wait, P_k is a 2x2 matrix of blocks.
                # The inner Strassen computes P_k.
                # P_k[I][J] is a 2x2 matrix of scalars.
                # The inner Strassen output C_coeffs[b_in][s] applies to the block b_in of P_k.
                # So the scalar value of the inner term s contributes to P_k[I][J] with coefficient C_coeffs[b_in][s] where b_in corresponds to (I,J).
                
                # Let's map b_in to (I,J):
                # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
                
                # For each final block C_block (0..3):
                # C_block is a 2x2 matrix of scalars.
                # Its value is sum over outer k of outer_C_coeffs[C_block][k] * P_k[C_block's position in 2x2 block matrix]
                
                # Let's iterate over the 4 blocks of the final 4x4 C.
                # C_block 0: C_00 (top-left 2x2)
                # C_block 1: C_01 (top-right 2x2)
                # C_block 2: C_10 (bottom-left 2x2)
                # C_block 3: C_11 (bottom-right 2x2)
                
                # For a specific final block C_block, and a specific outer step k:
                # The contribution is outer_C_coeffs[C_block][k] * P_k[C_block]
                # P_k[C_block] is a 2x2 matrix.
                # P_k[C_block] is computed by inner Strassen.
                # P_k[C_block][i][j] = sum_s C_coeffs[C_block][s] * (U_term_i * V_term_j)?
                # No.
                
                # Let's re-evaluate.
                # The inner Strassen algorithm computes the product of two 2x2 matrices.
                # Let X = U_k and Y = V_k.
                # The 7 scalar products are S_1..S_7.
                # The resulting matrix Z = X*Y has entries:
                # Z[0][0] = S_1 + S_4 - S_5 + S_7
                # Z[0][1] = S_3 + S_5
                # Z[1][0] = S_2 + S_4
                # Z[1][1] = S_1 - S_2 + S_3 + S_6
                
                # So for a fixed k, the 2x2 matrix P_k = U_k * V_k is:
                # P_k[0][0] = S_1 + S_4 - S_5 + S_7
                # ...
                
                # Each S_s is a scalar product of two linear forms U_term_s and V_term_s.
                # So P_k[I][J] is a linear combination of the 7 scalar products S_s.
                # P_k[I][J] = sum_s C_coeffs[I*2+J][s] * S_s
                
                # Now, the final C is:
                # C_block[c] = sum_k outer_C_coeffs[c][k] * P_k[c]
                # where c is the block index (0..3) corresponding to the position in the 2x2 block matrix.
                
                # So for a specific final block c, the 2x2 matrix of scalars is:
                # C_c = sum_k outer_C_coeffs[c][k] * P_k[c]
                # = sum_k outer_C_coeffs[c][k] * sum_s C_coeffs[c][s] * S_s
                # = sum_s (sum_k outer_C_coeffs[c][k] * C_coeffs[c][s]) * S_s
                
                # So for each final block c, the coefficient of the scalar product S_s (for outer step k) is:
                # coeff = outer_C_coeffs[c][k] * C_coeffs[c][s]
                
                # Wait, S_s depends on k. S_s is the s-th inner term for the k-th outer step.
                # So for term (k,s), the contribution to final block c is:
                # outer_C_coeffs[c][k] * C_coeffs[c][s]
                
                # This coefficient applies to the 2x2 matrix of scalars in block c.
                # So for each element (i,j) in the 2x2 block c:
                # W[c*4 + i*2 + j][term_idx] += outer_C_coeffs[c][k] * C_coeffs[c][s]
                
                # Let's implement this.
                
                for c in range(4):
                    coeff = outer_C_coeffs[c][k] * C_coeffs[c][s]
                    if coeff != 0:
                        # Block c corresponds to 4 elements in the 16-vector W.
                        # c=0: 0,1,4,5
                        # c=1: 2,3,6,7
                        # c=2: 8,9,12,13
                        # c=3: 10,11,14,15
                        ranges = [
                            [0,1,4,5],
                            [2,3,6,7],
                            [8,9,12,13],
                            [10,11,14,15]
                        ]
                        for idx in ranges[c]:
                            W[idx][term_idx] += coeff
                            
    return U, V, W
