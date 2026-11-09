# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses Laderman's algorithm (rank 49) for the 4x4x4 part
    and schoolbook (rank 16) for the 4x4x1 part.
    Total rank: 65.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Dimensions for the decomposition matrices
    dim_U = n * m  # 16
    dim_V = m * p  # 20
    dim_W = p * n  # 20
    
    rank = 65
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # We are building linear combinations.
        # Each term in the sum is U[r] * V[r] * W[r]
        # But here we are defining the columns of U, V, W.
        # So for a specific rank index r, we set U[u_idx][r], V[v_idx][r], W[w_idx][r].
        pass

    # We need to iterate through the 49 Laderman terms and 16 schoolbook terms.
    # Let's generate the Laderman terms first.
    
    # Strassen 2x2 linear forms
    # X, Y are 2x2 matrices.
    # M0 = (X00 + X11)(Y00 + Y11)
    # M1 = (X10 + X11)Y00
    # M2 = X00(Y01 - Y11)
    # M3 = X11(Y10 - Y00)
    # M4 = (X00 + X01)Y11
    # M5 = (X10 - X00)(Y00 + Y01)
    # M6 = (X01 - X11)(Y10 + Y11)
    
    # We represent a linear form on a 2x2 matrix X as a dict {(u,v): coeff}
    # Or a list of 4 coefficients.
    
    def strassen_terms():
        # Returns list of 7 tuples: (X_coeffs, Y_coeffs, result_signs)
        # result_signs maps Z indices to coefficients of M_s
        # Actually, we need to know how M_s contributes to Z.
        # Z00 = M0 + M3 - M4 + M6
        # Z01 = M2 + M4
        # Z10 = M1 + M3
        # Z11 = M0 - M1 + M2 + M5
        
        # We will generate the scalar multiplications.
        # For each of the 7 M_s, we have X_s and Y_s.
        
        terms = []
        
        # M0
        X0 = [(0,0,1), (1,1,1)]
        Y0 = [(0,0,1), (1,1,1)]
        terms.append((X0, Y0))
        
        # M1
        X1 = [(1,0,1), (1,1,1)]
        Y1 = [(0,0,1)]
        terms.append((X1, Y1))
        
        # M2
        X2 = [(0,0,1)]
        Y2 = [(0,1,1), (1,1,-1)]
        terms.append((X2, Y2))
        
        # M3
        X3 = [(1,1,1)]
        Y3 = [(1,0,1), (0,0,-1)]
        terms.append((X3, Y3))
        
        # M4
        X4 = [(0,0,1), (0,1,1)]
        Y4 = [(1,1,1)]
        terms.append((X4, Y4))
        
        # M5
        X5 = [(1,0,1), (0,0,-1)]
        Y5 = [(0,0,1), (0,1,1)]
        terms.append((X5, Y5))
        
        # M6
        X6 = [(0,1,1), (1,1,-1)]
        Y6 = [(1,0,1), (1,1,1)]
        terms.append((X6, Y6))
        
        return terms

    strassen_2x2 = strassen_terms()
    
    # Block level Strassen for 4x4
    # P0 = (A00 + A11)(B00 + B11)
    # P1 = (A10 + A11)B00
    # P2 = A00(B01 - B11)
    # P3 = A11(B10 - B00)
    # P4 = (A00 + A01)B11
    # P5 = (A10 - A00)(B00 + B01)
    # P6 = (A01 - A11)(B10 + B11)
    
    # A_ij is a 2x2 block.
    # We need to define the linear forms for the inputs to the 2x2 Strassen.
    
    def get_block_form(block_indices):
        # block_indices is a list of (I, J, coeff)
        # Returns a list of (u, v, coeff) for the 2x2 matrix
        form = []
        for I, J, c in block_indices:
            for u in range(2):
                for v in range(2):
                    form.append((u, v, c))
        return form

    # Define the 7 block products P_k inputs
    # X_k and Y_k are 2x2 matrices of blocks.
    
    P_defs = []
    
    # P0: X = A00 + A11, Y = B00 + B11
    X0_blocks = [((0,0), 1), ((1,1), 1)]
    Y0_blocks = [((0,0), 1), ((1,1), 1)]
    P_defs.append((X0_blocks, Y0_blocks))
    
    # P1: X = A10 + A11, Y = B00
    X1_blocks = [((1,0), 1), ((1,1), 1)]
    Y1_blocks = [((0,0), 1)]
    P_defs.append((X1_blocks, Y1_blocks))
    
    # P2: X = A00, Y = B01 - B11
    X2_blocks = [((0,0), 1)]
    Y2_blocks = [((0,1), 1), ((1,1), -1)]
    P_defs.append((X2_blocks, Y2_blocks))
    
    # P3: X = A11, Y = B10 - B00
    X3_blocks = [((1,1), 1)]
    Y3_blocks = [((1,0), 1), ((0,0), -1)]
    P_defs.append((X3_blocks, Y3_blocks))
    
    # P4: X = A00 + A01, Y = B11
    X4_blocks = [((0,0), 1), ((0,1), 1)]
    Y4_blocks = [((1,1), 1)]
    P_defs.append((X4_blocks, Y4_blocks))
    
    # P5: X = A10 - A00, Y = B00 + B01
    X5_blocks = [((1,0), 1), ((0,0), -1)]
    Y5_blocks = [((0,0), 1), ((0,1), 1)]
    P_defs.append((X5_blocks, Y5_blocks))
    
    # P6: X = A01 - A11, Y = B10 + B11
    X6_blocks = [((0,1), 1), ((1,1), -1)]
    Y6_blocks = [((1,0), 1), ((1,1), 1)]
    P_defs.append((X6_blocks, Y6_blocks))
    
    # Now, for each P_k, we apply Strassen 2x2.
    # Each P_k is a 2x2 matrix product.
    # Let X_k, Y_k be the 2x2 block matrices.
    # The elements of X_k are linear forms on A.
    # The elements of Y_k are linear forms on B.
    
    # We need to map the 2x2 elements (u,v) to global A indices.
    # A index for block (I,J) and sub-index (u,v) is:
    # i = I * 2 + u
    # j = J * 2 + v
    # global_idx = i * 4 + j
    
    # Similarly for B:
    # j = J * 2 + u  (Note: B is 4x4, blocks are 2x2)
    # k = K * 2 + v
    # global_idx = j * 4 + k (for the 4x4 part, k < 4)
    
    # And for C:
    # i = I * 2 + u
    # k = K * 2 + v
    # global_idx = k * 4 + i
    
    # We will iterate k in 0..6 (block products)
    # and s in 0..6 (Strassen terms within block product)
    
    rank_idx = 0
    
    for k in range(7):
        X_blocks, Y_blocks = P_defs[k]
        
        # Construct the 2x2 matrix of linear forms for X_k and Y_k
        # X_k[u, v] is a linear form on A.
        # Y_k[u, v] is a linear form on B.
        
        X_forms = [[None for _ in range(2)] for _ in range(2)]
        Y_forms = [[None for _ in range(2)] for _ in range(2)]
        
        # Initialize with zeros
        for u in range(2):
            for v in range(2):
                X_forms[u][v] = {}
                Y_forms[u][v] = {}
        
        for (I, J), c in X_blocks:
            for u in range(2):
                for v in range(2):
                    # A index
                    i = I * 2 + u
                    j = J * 2 + v
                    a_idx = i * 4 + j
                    X_forms[u][v][a_idx] = X_forms[u][v].get(a_idx, 0) + c
        
        for (I, J), c in Y_blocks:
            for u in range(2):
                for v in range(2):
                    # B index
                    j = I * 2 + u
                    k = J * 2 + v
                    b_idx = j * 4 + k
                    Y_forms[u][v][b_idx] = Y_forms[u][v].get(b_idx, 0) + c
        
        # Now apply Strassen 2x2 to X_k, Y_k
        # The Strassen terms M_s are products of linear forms.
        # For each M_s, we get a scalar multiplication.
        
        for s in range(7):
            X_s_coeffs, Y_s_coeffs = strassen_2x2[s]
            
            # X_s is a linear form on the 2x2 matrix X_k.
            # X_s = sum_{u,v} coeff_{u,v} * X_k[u,v]
            # Y_s = sum_{u,v} coeff_{u,v} * Y_k[u,v]
            
            # Combine to get linear forms on A and B.
            U_form = {}
            V_form = {}
            
            for (u, v, c) in X_s_coeffs:
                for a_idx, ac in X_forms[u][v].items():
                    U_form[a_idx] = U_form.get(a_idx, 0) + c * ac
            
            for (u, v, c) in Y_s_coeffs:
                for b_idx, bc in Y_forms[u][v].items():
                    V_form[b_idx] = V_form.get(b_idx, 0) + c * bc
            
            # Now we have a term: (sum U_form[a] A[a]) * (sum V_form[b] B[b])
            # This contributes to the result C.
            
            # We need to determine the contribution to W.
            # The product is M_s of P_k.
            # P_k is a 2x2 matrix.
            # M_s contributes to P_k[0,0], P_k[0,1], etc.
            
            # From Strassen 2x2:
            # Z00 = M0 + M3 - M4 + M6
            # Z01 = M2 + M4
            # Z10 = M1 + M3
            # Z11 = M0 - M1 + M2 + M5
            
            # So for a given s, we know which Z elements it affects and with what sign.
            # Let's create a mapping s -> list of (z_u, z_v, sign)
            
            contribs = []
            if s == 0:
                contribs = [(0,0,1), (1,1,1)]
            elif s == 1:
                contribs = [(1,0,1), (1,1,-1)]
            elif s == 2:
                contribs = [(0,1,1), (1,1,1)]
            elif s == 3:
                contribs = [(0,0,1), (1,0,1)]
            elif s == 4:
                contribs = [(0,0,-1), (0,1,1)]
            elif s == 5:
                contribs = [(1,1,1)]
            elif s == 6:
                contribs = [(0,0,1)]
            
            # Now, P_k contributes to C blocks.
            # C00 = P0 + P3 - P4 + P6
            # C01 = P2 + P4
            # C10 = P1 + P3
            # C11 = P0 - P1 + P2 + P5
            
            # So for P_k, we have a mapping k -> list of (C_I, C_J, sign)
            
            P_contribs = []
            if k == 0:
                P_contribs = [(0,0,1), (1,1,1)]
            elif k == 1:
                P_contribs = [(1,0,1), (1,1,-1)]
            elif k == 2:
                P_contribs = [(0,1,1), (1,1,1)]
            elif k == 3:
                P_contribs = [(0,0,1), (1,0,1)]
            elif k == 4:
                P_contribs = [(0,0,-1), (0,1,1)]
            elif k == 5:
                P_contribs = [(1,1,1)]
            elif k == 6:
                P_contribs = [(0,0,1)]
            
            # Total contribution to C[I_C, J_C] from this term is:
            # sum_{z_u, z_v} (sign_z) * sum_{C_I, C_J} (sign_C) * delta(I_C == z_u, J_C == z_v) ?
            # No.
            # P_k is a 2x2 matrix.
            # M_s contributes to P_k[z_u, z_v] with sign_z.
            # P_k contributes to C[I_C, J_C] with sign_C if (I_C, J_C) matches the block position?
            # Wait.
            # C is a 4x4 matrix.
            # C[I_C, J_C] is a 2x2 block.
            # The formula C00 = P0 + ... means the block C00 is the sum of the matrices P0, etc.
            # So P_k[z_u, z_v] is the element (z_u, z_v) of the block P_k.
            # This element adds to C[I_C, J_C][z_u, z_v] with sign_C.
            
            # So for each (z_u, z_v, sign_z) in contribs:
            #   For each (I_C, J_C, sign_C) in P_contribs:
            #       The element C[I_C * 2 + z_u, J_C * 2 + z_v] gets contribution sign_z * sign_C.
            
            W_form = {}
            for (z_u, z_v, sign_z) in contribs:
                for (I_C, J_C, sign_C) in P_contribs:
                    i = I_C * 2 + z_u
                    k = J_C * 2 + z_v
                    w_idx = k * 4 + i
                    W_form[w_idx] = W_form.get(w_idx, 0) + sign_z * sign_C
            
            # Now we have U_form, V_form, W_form for this term.
            # Add to U, V, W matrices at column rank_idx.
            
            for a_idx, val in U_form.items():
                U[a_idx][rank_idx] += val
            for b_idx, val in V_form.items():
                V[b_idx][rank_idx] += val
            for w_idx, val in W_form.items():
                W[w_idx][rank_idx] += val
            
            rank_idx += 1
    
    # Now add the 4x4x1 part (schoolbook)
    # A[i,j] * B[j,4] = C[i,4]
    for i in range(4):
        for j in range(4):
            a_idx = i * 4 + j
            b_idx = j * 5 + 4  # p=5, so last col is 4
            w_idx = 4 * 4 + i  # k=4, n=4, so 4*4+i = 16+i
            
            U[a_idx][rank_idx] = 1
            V[b_idx][rank_idx] = 1
            W[w_idx][rank_idx] = 1
            rank_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
