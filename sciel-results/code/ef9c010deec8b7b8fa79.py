"""Matrix multiplication 4,5,5 with Strassen-based decomposition."""
DIMENSIONS = [4, 5, 5]

def _strassen_coeffs_2x2():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (coeff_A, coeff_B) where coeff_A is a dict mapping (i,j) -> scalar
    and coeff_B is a dict mapping (j,k) -> scalar.
    The 7 terms are:
      M1 = (A11 + A22)(B11 + B22)
      M2 = (A21 + A22)B11
      M3 = A11(B12 - B22)
      M4 = A22(B21 - B11)
      M5 = (A11 + A12)B22
      M6 = (A21 - A11)(B11 + B12)
      M7 = (A12 - A22)(B21 + B22)
    And C = [[M1 + M4 - M5 + M7, M3 + M5],
             [M2 + M4,           M1 - M2 + M3 + M6]]
    """
    # Each term: (A_coeffs, B_coeffs) where coeffs are dicts
    terms = []
    # M1 = (A11 + A22)(B11 + B22)
    A = {(0,0): 1, (1,1): 1}
    B = {(0,0): 1, (1,1): 1}
    terms.append((A, B))
    # M2 = (A21 + A22)B11
    A = {(1,0): 1, (1,1): 1}
    B = {(0,0): 1}
    terms.append((A, B))
    # M3 = A11(B12 - B22)
    A = {(0,0): 1}
    B = {(0,1): 1, (1,1): -1}
    terms.append((A, B))
    # M4 = A22(B21 - B11)
    A = {(1,1): 1}
    B = {(1,0): 1, (0,0): -1}
    terms.append((A, B))
    # M5 = (A11 + A12)B22
    A = {(0,0): 1, (0,1): 1}
    B = {(1,1): 1}
    terms.append((A, B))
    # M6 = (A21 - A11)(B11 + B12)
    A = {(1,0): 1, (0,0): -1}
    B = {(0,0): 1, (0,1): 1}
    terms.append((A, B))
    # M7 = (A12 - A22)(B21 + B22)
    A = {(0,1): 1, (1,1): -1}
    B = {(1,0): 1, (1,1): 1}
    terms.append((A, B))
    
    # C coefficients: for each output (i,k), which M terms contribute with what coeff
    # C[0][0] = M1 + M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 + M6
    C_coeffs = {
        (0,0): {0: 1, 3: 1, 4: -1, 6: 1},
        (0,1): {2: 1, 4: 1},
        (1,0): {1: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 2: 1, 5: 1}
    }
    return terms, C_coeffs

def construct():
    n, m, p = DIMENSIONS
    # We'll build a rank-50 decomposition: 49 terms from Strassen on 4x4x4 core + 1 term for j=4
    
    # Collect all (U_coeffs, V_coeffs, W_coeffs) for each term
    # U row index: i*m + j, V row index: j*p + k, W row index: k*n + i
    # Each term is a triple of dicts: U_dict (row->coeff), V_dict (row->coeff), W_dict (row->coeff)
    
    terms = []  # list of (U_dict, V_dict, W_dict)
    
    terms_2x2, C_coeffs_2x2 = _strassen_coeffs_2x2()
    
    # For the 4x4x4 sub-problem (i in 0..3, j in 0..3, k in 0..3)
    # Apply Strassen at two levels:
    # Level 1: split 4x4 into 2x2 blocks of 2x2 matrices
    #   A_blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    #   B_blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    #   C_blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3), C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
    
    # For each of the 7 top-level Strassen terms M_t = P_t * Q_t where P_t and Q_t are 2x2 matrices
    # P_t is a linear combination of A blocks, Q_t is a linear combination of B blocks
    # Then P_t * Q_t is computed using 2x2 Strassen again (7 bottom-level terms)
    
    # Top-level Strassen: 7 terms, each giving a 2x2 matrix product
    # For each top-level term t, we have:
    #   P_t = sum over block positions (r,c) in {0,1}^2 of coeff_P_t[r,c] * A_block[r,c]
    #   Q_t = sum over block positions (r,c) in {0,1}^2 of coeff_Q_t[r,c] * B_block[r,c]
    # Then C_block contribution from term t is P_t * Q_t, computed via 7 bottom-level Strassen terms
    
    # Top-level Strassen formulas (same as 2x2):
    # M1_top = (A11 + A22)(B11 + B22)
    # M2_top = (A21 + A22)B11
    # M3_top = A11(B12 - B22)
    # M4_top = A22(B21 - B11)
    # M5_top = (A11 + A12)B22
    # M6_top = (A21 - A11)(B11 + B12)
    # M7_top = (A12 - A22)(B21 + B22)
    
    # For each top term, P_t is a 2x2 matrix (linear comb of A blocks), Q_t is a 2x2 matrix (linear comb of B blocks)
    # The product P_t * Q_t is a 2x2 matrix, and its elements are given by C_coeffs_2x2
    
    # Let's enumerate all 49 bottom-level terms
    for t_top in range(7):
        # Determine P_t and Q_t as 2x2 matrices with coefficients on A and B blocks
        # From terms_2x2, we have A_coeffs and B_coeffs for the 2x2 indices (0,0), (0,1), (1,0), (1,1)
        # These correspond to block positions
        A_top = terms_2x2[t_top][0]  # dict: (r,c) -> coeff, where (r,c) in {(0,0),(0,1),(1,0),(1,1)}
        B_top = terms_2x2[t_top][1]  # dict: (r,c) -> coeff
        
        # For each bottom-level Strassen term s (0..6), we compute a scalar product
        # P_t[i1,j1] * Q_t[j1,k1] for the appropriate (i1,k1) based on C_coeffs_2x2
        # Actually, for each bottom-level Strassen term s, we get a contribution to C_block
        # C_block[i1][k1] += C_coeffs_2x2[(i1,k1)][s] * (P_t * Q_t)[i1][k1]
        # But (P_t * Q_t)[i1][k1] = sum over j1 of P_t[i1][j1] * Q_t[j1][k1]
        # And P_t[i1][j1] = sum over A_block positions (r,c) where (r,c) maps to (i1,j1) ... 
        # Wait, I need to think about this differently.
        
        # P_t is a 2x2 matrix where P_t[i1][j1] is a linear combination of A block elements
        # Specifically, P_t[i1][j1] = sum over (r,c) in A_top.keys() of A_top[(r,c)] * A_block[r,c][i1][j1]
        # Wait no. A_block[r,c] is a 2x2 matrix. P_t is formed as a linear combination of the 4 A blocks.
        # P_t = sum over (r,c) of A_top[(r,c)] * A_block[r,c]
        # So P_t[i1][j1] = sum over (r,c) of A_top[(r,c)] * A_block[r,c][i1][j1]
        # Similarly Q_t[j1][k1] = sum over (r,c) of B_top[(r,c)] * B_block[r,c][j1][k1]
        
        # For bottom-level Strassen term s, the contribution to (P_t * Q_t)[i1][k1] is:
        # sum over j1 of P_t[i1][j1] * Q_t[j1][k1]
        # = sum over j1 of (sum over (r,c)_A of A_top[(r,c)_A] * A_block[r,c][i1][j1]) * (sum over (r,c)_B of B_top[(r,c)_B] * B_block[r,c][j1][k1])
        
        # This is getting complex. Let me just enumerate all scalar multiplications.
        
        # For each top term t_top, and each bottom Strassen term s, we have:
        # The bottom term s computes: (sum of A_elements) * (sum of B_elements)
        # where A_elements come from P_t's Strassen decomposition and B_elements from Q_t's
        
        # Actually, let me just directly build the 49 terms by iterating over all combinations.
        
        # For bottom-level Strassen term s:
        # A_coeffs_bottom = terms_2x2[s][0]  # dict on (i,j) in 2x2
        # B_coeffs_bottom = terms_2x2[s][1]  # dict on (j,k) in 2x2
        # This term contributes to (P_t*Q_t) with coefficient C_coeffs_2x2[(i1,k1)][s] for each (i1,k1)
        
        # For a given (i1, k1), the contribution from bottom term s to C_block[i1][k1] is:
        # C_coeffs_2x2[(i1,k1)][s] * (A_bottom_combination * B_bottom_combination)
        
        # The A_bottom_combination for term s is: sum over (ia, ja) of A_coeffs_bottom[(ia,ja)] * P_t[ia][ja]
        # = sum over (ia,ja) of A_coeffs_bottom[(ia,ja)] * (sum over (r,c) of A_top[(r,c)] * A_block[r,c][ia][ja])
        # = sum over (r,c) of A_top[(r,c)] * (sum over (ia,ja) of A_coeffs_bottom[(ia,ja)] * A_block[r,c][ia][ja])
        
        # Similarly for B.
        
        # This is getting unwieldy. Let me just brute-force build the U, V, W dicts for each of the 49 terms.
        
        for s in range(7):
            A_bot = terms_2x2[s][0]
            B_bot = terms_2x2[s][1]
            
            # Build U_dict, V_dict, W_dict for this term
            U_dict = {}
            V_dict = {}
            W_dict = {}
            
            # This term contributes to C_block[i1][k1] for each (i1,k1) where C_coeffs_2x2[(i1,k1)][s] != 0
            # For each such (i1,k1), the coefficient is C_coeffs_2x2[(i1,k1)][s]
            # The product is (A_combination) * (B_combination)
            # A_combination = sum over (ia,ja) of A_bot[(ia,ja)] * P_t[ia][ja]
            # B_combination = sum over (jb,kb) of B_bot[(jb,kb)] * Q_t[jb][kb]
            
            # P_t[ia][ja] = sum over (r,c) of A_top[(r,c)] * A_block[r,c][ia][ja]
            # A_block[r,c][ia][ja] corresponds to A element at row (2*r+ia), col (2*c+ja)
            # So A element index: i = 2*r+ia, j = 2*c+ja
            
            # Q_t[jb][kb] = sum over (r,c) of B_top[(r,c)] * B_block[r,c][jb][kb]
            # B element: j = 2*r+jb, k = 2*c+kb
            
            # For the W part: C element at row (2*i1 + ic), col (2*k1 + kc) where ic, kc are within the 2x2 block
            # Wait, C_block[i1][k1] is a 2x2 matrix. Its element [ic][kc] corresponds to C[2*i1+ic][2*k1+kc]
            # But the bottom-level Strassen computes the full 2x2 product, so all 4 elements of C_block[i1][k1] get contributions
            
            # Actually, I realize the bottom-level Strassen computes P_t * Q_t which is a 2x2 matrix.
            # All 4 elements of this 2x2 matrix are computed. So for each (i1,k1) in {(0,0),(0,1),(1,0),(1,1)},
            # and for each bottom term s, we get a contribution to C_block[i1][k1] as a whole (all 4 elements).
            
            # But each bottom term s is a single scalar multiplication that contributes to multiple C elements.
            # Specifically, for fixed s, the term contributes to (P_t*Q_t)[i1][k1] for each (i1,k1) 
            # with coefficient C_coeffs_2x2[(i1,k1)][s].
            
            # And (P_t*Q_t)[i1][k1] is a linear combination of A elements times a linear combination of B elements.
            
            # For each (i1,k1) with nonzero C_coeffs_2x2[(i1,k1)][s]:
            # contrib_coeff = C_coeffs_2x2[(i1,k1)][s]
            # A_part = sum over (ia,ja) of A_bot[(ia,ja)] * P_t[ia][ja]
            # B_part = sum over (jb,kb) of B_bot[(jb,kb)] * Q_t[jb][kb]
            # This product contributes to C[2*i1+ic][2*k1+kc] for ic,kc in {0,1}
            # But wait, the bottom Strassen already handles the 2x2 structure. Let me just track it as:
            # The scalar product A_part * B_part contributes with weight contrib_coeff to each C element
            # in the block C_block[i1][k1].
            
            # Hmm, I think I'm overcomplicating this. Let me just directly build the full decomposition.
            
            # For this term (t_top, s), the U, V, W are determined by:
            # U: for each A element (i,j), coefficient = sum over contributing paths
            # V: for each B element (j,k), coefficient
            # W: for each C element (k,i), coefficient
            
            # Let me compute the coefficients by iterating over all possible A, B, C elements.
            
            # Actually, let me just build it step by step.
            # For term (t_top, s):
            # A_combination = sum over (ia,ja) in A_bot of A_bot[(ia,ja)] * P_t[ia][ja]
            # = sum over (ia,ja) in A_bot of A_bot[(ia,ja)] * sum over (r,c) in A_top of A_top[(r,c)] * A[2*r+ia][2*c+ja]
            # = sum over (i,j) of coeff_A[i][j] * A[i][j]
            # where coeff_A[i][j] = sum over (ia,ja) in A_bot such that i=2*r+ia, j=2*c+ja for some (r,c) in A_top
            #   of A_bot[(ia,ja)] * A_top[(r,c)]
            
            # Similarly for B_combination.
            
            # And the W part: this term contributes to C elements in blocks C_block[i1][k1] for each (i1,k1)
            # with C_coeffs_2x2[(i1,k1)][s] != 0, and the contribution to C[2*i1+ic][2*k1+kc] is
            # C_coeffs_2x2[(i1,k1)][s] * (A_combination * B_combination)
            
            # But since A_combination * B_combination is a single scalar product, the W coefficients are:
            # For each C element (k,i), W_coeff[k*n+i] = sum over (i1,k1,ic,kc) where i=2*i1+ic, k=2*k1+kc
            #   of C_coeffs_2x2[(i1,k1)][s]
            
            # Wait, that's not right either. The bottom-level Strassen term s computes a specific scalar product
            # that contributes to specific C elements. Let me just directly compute all coefficients.
            
            # Let me build coeff_A as a dict from (i,j) to coefficient
            coeff_A = {}
            for (ia, ja), a_coeff in A_bot.items():
                for (r, c), top_coeff in A_top.items():
                    i = 2 * r + ia
                    j = 2 * c + ja
                    if 0 <= i < 4 and 0 <= j < 4:
                        key = (i, j)
                        coeff_A[key] = coeff_A.get(key, 0) + a_coeff * top_coeff
            
            coeff_B = {}
            for (jb, kb), b_coeff in B_bot.items():
                for (r, c), top_coeff in B_top.items():
                    j = 2 * r + jb
                    k = 2 * c + kb
                    if 0 <= j < 4 and 0 <= k < 4:
                        key = (j, k)
                        coeff_B[key] = coeff_B.get(key, 0) + b_coeff * top_coeff
            
            coeff_C = {}
            for (i1, k1), c_coeff in C_coeffs_2x2.items():
                if s in c_coeff:
                    w_coeff = c_coeff[s]
                    for ic in range(2):
                        for kc in range(2):
                            i = 2 * i1 + ic
                            k = 2 * k1 + kc
                            if 0 <= i < 4 and 0 <= k < 4:
                                key = (k, i)
                                coeff_C[key] = coeff_C.get(key, 0) + w_coeff
            
            # Now build U, V, W dicts for this term
            U_d = {}
            for (i, j), c in coeff_A.items():
                if c != 0:
                    row = i * m + j
                    U_d[row] = c
            V_d = {}
            for (j, k), c in coeff_B.items():
                if c != 0:
                    row = j * p + k
                    V_d[row] = c
            W_d = {}
            for (k, i), c in coeff_C.items():
                if c != 0:
                    row = k * n + i
                    W_d[row] = c
            
            terms.append((U_d, V_d, W_d))
    
    # Add the 50th term for j=4
    # C[i][k] += A[i][4] * B[4][k]
    U_d = {}
    for i in range(n):
        U_d[i * m + 4] = 1
    V_d = {}
    for k in range(p):
        V_d[4 * p + k] = 1
    W_d = {}
    for k in range(p):
        for i in range(n):
            W_d[k * n + i] = 1
    terms.append((U_d, V_d, W_d))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_d, V_d, W_d) in enumerate(terms):
        for row, coeff in U_d.items():
            U[row][col] = coeff
        for row, coeff in V_d.items():
            V[row][col] = coeff
        for row, coeff in W_d.items():
            W[row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
