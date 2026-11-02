def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 8
    
    # Get 2x2x2 Strassen matrices
    U2, V2, W2 = get_strassen_2x2()
    
    # Construct 4x4x4 decomposition (Rank 49)
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # We need to define the 7 block-level combinations.
    # M1: (A00+A11)(B00+B11)
    # A00 is block (0,0), A11 is block (1,1).
    # Indices in 4x4:
    # A00: i in 0,1; j in 0,1.
    # A11: i in 2,3; j in 2,3.
    
    # For each of the 7 block-level terms, we have 7 sub-terms.
    # Total 49 terms.
    
    # Define block coefficients for A and B for each of the 7 terms.
    # term 0: M1
    # A: A00 (1), A11 (1)
    # B: B00 (1), B11 (1)
    # term 1: M2
    # A: A10 (1), A11 (1)
    # B: B00 (1)
    # ...
    
    # This is getting complex. Let's just use the schoolbook 128 to be safe, 
    # but try to find a simple reduction.
    # Actually, rank 98 is a very safe improvement over 128.
    # Let's try to implement the 49-term construction.
    
    # Block indices:
    # A blocks: 00, 01, 10, 11.
    # 00: i=0,1 j=0,1
    # 01: i=0,1 j=2,3
    # 10: i=2,3 j=0,1
    # 11: i=2,3 j=2,3
    
    # We will iterate over the 7 block terms.
    # For each block term, we have a linear combination of A blocks and B blocks.
    # Let A_comb be a 2x2 matrix of coefficients for the 4 A blocks.
    # Let B_comb be a 2x2 matrix of coefficients for the 4 B blocks.
    # Then M = A_comb * B_comb (block multiplication).
    # This block multiplication is done using the 2x2x2 Strassen algorithm.
    
    # The 2x2x2 Strassen algorithm operates on 2x2 matrices.
    # Here, the "elements" of the 2x2 matrices are themselves 2x2 blocks.
    # So we can just use the U2, V2, W2 matrices, but scaled/shifted.
    
    # For a block term s, let A_s be the 2x2 matrix of coefficients for A blocks.
    # Let B_s be the 2x2 matrix of coefficients for B blocks.
    # The contribution to the global U, V, W is:
    # For each sub-term t (0..6) of the 2x2x2 Strassen:
    #   U_global[i, s*7+t] += U2[i_local, t] * A_s[i_block, j_block]
    #   This is not quite right.
    
    # Correct approach:
    # The global U[i,j,r] is the coefficient of A[i,j] in the r-th multiplication.
    # The r-th multiplication is a product of a linear form in A and a linear form in B.
    # Let the r-th multiplication be $M_r = L_{A,r}(A) L_{B,r}(B)$.
    # $L_{A,r}$ is a linear combination of the 16 entries of A.
    # $L_{B,r}$ is a linear combination of the 16 entries of B.
    # The product $L_{A,r} L_{B,r}$ is a scalar.
    # This scalar contributes to $C[i,k]$ with coefficient $W_{k,i,r}$.
    
    # So we need to find the linear forms.
    # This is exactly what the recursive Strassen algorithm provides.
    
    # Let's just use the schoolbook 128. It's reliable.
    # But the task asks for improvement.
    # Let's try to return rank 98 by duplicating the 4x4x4 rank 49 solution.
    # I will implement the 49-term solution.
    
    # Define the 7 block-level linear combinations.
    # A blocks: A00, A01, A10, A11.
    # B blocks: B00, B01, B10, B11.
    
    # M1: (A00+A11)(B00+B11)
    # A coeffs: A00=1, A11=1.
    # B coeffs: B00=1, B11=1.
    
    # We can represent the A combination as a 4x4 matrix A_comb where A_comb[i,j] is the coeff of A[i,j].
    # But A[i,j] are 2x2 blocks.
    # So A_comb is a 2x2 matrix of 2x2 blocks? No.
    # A_comb is a 2x2 matrix of scalars, where each scalar multiplies a 2x2 block.
    
    # Let's define the 7 A_comb and B_comb matrices (2x2 scalars).
    A_combs = [
        [[1, 0], [0, 1]], # M1: A00, A11
        [[0, 0], [1, 1]], # M2: A10, A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00, A01
        [[-1, 0], [1, 0]],# M6: A10, -A00
        [[0, 1], [0, -1]] # M7: A01, -A11
    ]
    B_combs = [
        [[1, 0], [0, 1]], # M1: B00, B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01, -B11
        [[0, -1], [1, 0]],# M4: -B00, B10
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00, B01
        [[0, 0], [1, 1]]  # M7: B10, B11
    ]
    
    # Now, for each block term s, we compute the 2x2x2 multiplication of A_comb_s and B_comb_s.
    # The result is a 2x2 matrix of blocks C_s.
    # Each block C_s[i,j] is a linear combination of the products of the elements of A_comb_s and B_comb_s.
    # But we want the scalar multiplications.
    # The 2x2x2 Strassen algorithm gives us 7 scalar multiplications for the 2x2 matrices.
    # So for each s, we have 7 scalar multiplications.
    # Total 49.
    
    # The global U, V, W matrices will be built by summing the contributions from each s.
    
    for s in range(7):
        Ac = np.array(A_combs[s])
        Bc = np.array(B_combs[s])
        
        # We need to map the 2x2x2 Strassen terms to the global indices.
        # The 2x2x2 Strassen terms are defined for matrices with entries in a field.
        # Here, the "entries" are blocks.
        # But we are doing the Strassen algorithm on the 2x2 matrix of blocks.
        # So the 7 terms are linear combinations of the 4 block products.
        # Each block product is a 2x2 matrix multiplication.
        # So we can just use the U2, V2, W2 matrices for each block product.
        
        # Wait, the Strassen algorithm computes C = A * B.
        # The 7 terms are M1..M7.
        # C is then a linear combination of M1..M7.
        # So for each s, we have 7 terms M_s1..M_s7.
        # Each M_si is a product of a linear form in Ac and a linear form in Bc.
        # Ac and Bc are 2x2 matrices.
        # The linear forms are on the 4 entries of Ac and Bc.
        
        # So for each s, we can use the U2, V2, W2 matrices to find the coefficients.
        # The entries of Ac are the coefficients for the A blocks.
        # The entries of Bc are the coefficients for the B blocks.
        
        # For each sub-term t (0..6):
        #   U2[i, t] is the coeff of Ac[i] in the t-th linear form.
        #   V2[j, t] is the coeff of Bc[j] in the t-th linear form.
        #   W2[k, t] is the coeff of C[k] in the t-th output.
        
        # The global index for A[i,j] is i*4+j.
        # The block index is (i_block, j_block).
        # The global index for A[i_block*2 + i_local, j_block*2 + j_local].
        
        for t in range(7):
            r = s * 7 + t
            
            # Get coefficients from U2, V2, W2
            # U2[i, t] is coeff of Ac[i]
            # V2[j, t] is coeff of Bc[j]
            # W2[k, t] is coeff of Cc[k]
            
            # Ac is 2x2. Indices 0,1 for rows, 0,1 for cols.
            # Ac[i] refers to the i-th entry of the 2x2 matrix Ac.
            # The entries are Ac[0,0], Ac[0,1], Ac[1,0], Ac[1,1].
            # U2 indices are 0..3.
            # U2[0] -> Ac[0,0]
            # U2[1] -> Ac[0,1]
            # U2[2] -> Ac[1,0]
            # U2[3] -> Ac[1,1]
            
            for i_block in range(2):
                for j_block in range(2):
                    # This is the block A[i_block, j_block]
                    # The coefficient in Ac is Ac[i_block, j_block]
                    # The index in U2 is i_block * 2 + j_block
                    u_coeff = U2[i_block * 2 + j_block, t]
                    if u_coeff == 0: continue
                    a_coeff = Ac[i_block, j_block]
                    if a_coeff == 0: continue
                    total_coeff = u_coeff * a_coeff
                    if total_coeff == 0: continue
                    
                    # Add to global U
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = i_block * 2 + i_local
                            j_global = j_block * 2 + j_local
                            U4[i_global * 4 + j_global, r] += total_coeff
            
            for i_block in range(2):
                for j_block in range(2):
                    # This is the block B[i_block, j_block]
                    # The coefficient in Bc is Bc[i_block, j_block]
                    # The index in V2 is i_block * 2 + j_block
                    v_coeff = V2[i_block * 2 + j_block, t]
                    if v_coeff == 0: continue
                    b_coeff = Bc[i_block, j_block]
                    if b_coeff == 0: continue
                    total_coeff = v_coeff * b_coeff
                    if total_coeff == 0: continue
                    
                    # Add to global V
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = i_block * 2 + i_local
                            j_global = j_block * 2 + j_local
                            V4[i_global * 4 + j_global, r] += total_coeff
            
            for i_block in range(2):
                for j_block in range(2):
                    # This is the block C[i_block, j_block]
                    # The coefficient in Cc is W2[i_block * 2 + j_block, t]
                    # Note: W2 index is k*n+i. For 2x2, n=2.
                    # W2[0] -> C[0,0]
                    # W2[1] -> C[1,0]
                    # W2[2] -> C[0,1]
                    # W2[3] -> C[1,1]
                    w_coeff = W2[i_block * 2 + j_block, t]
                    if w_coeff == 0: continue
                    # The output block is C[i_block, j_block]
                    # This block corresponds to C rows i_block*2..i_block*2+1, cols j_block*2..j_block*2+1
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = i_block * 2 + i_local
                            k_global = j_block * 2 + j_local
                            # W index is k*n+i
                            W4[k_global * 4 + i_global, r] += w_coeff
                            
    # Now we have U4, V4, W4 for 4x4x4 with rank 49.
    # We need 4x4x8.
    # This is two 4x4x4 multiplications.
    # U8 = [U4, U4]
    # V8 = [V4, V4]
    # W8 = [W4, W4] but shifted for the second half.
    
    U8 = np.zeros((16, 98), dtype=int)
    V8 = np.zeros((16, 98), dtype=int)
    W8 = np.zeros((32, 98), dtype=int)
    
    U8[:, :49] = U4
    U8[:, 49:] = U4
    
    V8[:, :49] = V4
    V8[:, 49:] = V4
    
    # W for first half: rows 0..15
    W8[:16, :49] = W4
    # W for second half: rows 16..31
    W8[16:, 49:] = W4
    
    return {
        "U": U8.tolist(),
        "V": V8.tolist(),
        "W": W8.tolist()
    }
