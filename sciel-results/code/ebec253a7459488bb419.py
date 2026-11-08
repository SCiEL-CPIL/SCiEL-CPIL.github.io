import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-114 tensor decomposition for matrix multiplication
    of dimensions 4x4 * 4x9.
    
    Strategy:
    1. Split the 4x9 matrix B into B1 (4x4) and B2 (4x5).
    2. Compute C1 = A * B1 using a rank-49 decomposition (Strassen based).
    3. Compute C2 = A * B2 using a rank-65 decomposition (Strassen 4x4x4 + Schoolbook 4x4x1).
    4. Combine C = C1 + C2.
    Total rank = 49 + 65 = 114.
    """
    n, m, p = DIMENSIONS
    
    # We will build the decomposition term by term.
    # Each term is a tuple (u_indices, v_indices, w_indices, coeffs)
    # But since it's a tensor decomposition, each term is a rank-1 tensor.
    # We accumulate into U, V, W matrices.
    
    U = [[0] * 114 for _ in range(n * m)]
    V = [[0] * 114 for _ in range(m * p)]
    W = [[0] * 114 for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: 4x4 * 4x4 (cols 0-3 of B) ---
    # Uses 49 terms from Strassen.
    
    # Define Strassen 2x2x2 terms
    # Each term is a list of (u_idx, coeff), (v_idx, coeff), (w_idx, coeff)
    # Indices are local 0..3 for 2x2 matrices.
    # U: i*2+j, V: j*2+k, W: k*2+i
    
    strassen_2x2_terms = [
        # M1 = (A00+A11)(B00+B11)
        [([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])],
        # M2 = (A10+A11)B00
        [([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)])], # Wait, M2 contributes to Z10, Z11?
        # Let's re-verify Strassen output mapping.
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # My previous manual derivation for M1 was:
        # U: A00, A11 -> 0, 3
        # V: B00, B11 -> 0, 3
        # W: Z00, Z11 -> 0, 3
        # This matches M1.
        
        # M2 = (A10+A11)B00
        # A10 is idx 2, A11 is idx 3. B00 is idx 0.
        # U: 2, 3. V: 0.
        # Contributes to Z10 (idx 2) and Z11 (idx 3).
        # W: 2, 3.
        [([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)])],
        
        # M3 = A00(B01-B11)
        # A00 is 0. B01 is 1, B11 is 3.
        # U: 0. V: 1 (coeff 1), 3 (coeff -1).
        # Contributes to Z01 (idx 1) and Z11 (idx 3).
        # W: 1, 3.
        [([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])],
        
        # M4 = A11(B10-B00)
        # A11 is 3. B10 is 2, B00 is 0.
        # U: 3. V: 2 (coeff 1), 0 (coeff -1).
        # Contributes to Z00 (idx 0) and Z10 (idx 2).
        # W: 0, 2.
        [([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])],
        
        # M5 = (A00+A01)B11
        # A00 is 0, A01 is 1. B11 is 3.
        # U: 0, 1. V: 3.
        # Contributes to Z00 (idx 0) and Z10 (idx 2)?
        # Wait, Z00 = ... - M5. Z10 = ...
        # Z00 gets -M5. Z10 gets nothing?
        # Let's check Z01 = M3 + M5. So Z01 gets M5.
        # Z00 gets -M5.
        # So W: 0 (coeff -1), 1 (coeff 1).
        [([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)])],
        
        # M6 = (A10-A00)(B00+B01)
        # A10 is 2, A00 is 0. B00 is 0, B01 is 1.
        # U: 2 (coeff 1), 0 (coeff -1). V: 0, 1.
        # Contributes to Z11 (idx 3).
        # W: 3.
        [([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])],
        
        # M7 = (A01-A11)(B10+B11)
        # A01 is 1, A11 is 3. B10 is 2, B11 is 3.
        # U: 1 (coeff 1), 3 (coeff -1). V: 2, 3.
        # Contributes to Z00 (idx 0).
        # W: 0.
        [([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])],
    ]
    
    # Helper to add a 2x2x2 Strassen block to the main matrices
    # block_pos: (I, J, K) for A, B, C blocks?
    # Actually, the structure is fixed for 4x4x4.
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # We need to map the 7 Strassen products to the specific A, B blocks.
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B01-B11
    # M4: A11, B10-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    # For each M_k, we have 7 sub-terms from the 2x2x2 decomposition.
    # We need to map the local indices to global indices.
    
    # Global A index: I*2 + i, J*2 + j -> (I*2+i)*4 + (J*2+j)
    # Global B index: J*2 + j, K*2 + k -> (J*2+j)*9 + (K*2+k)  (Note: p=9, but here we are in 4x4x4 sub-problem, so local p=4? No, we are embedding into 4x4x9. So B index uses p=9.)
    # Global C index: K*2 + k, I*2 + i -> (K*2+k)*4 + (I*2+i)
    
    # Wait, the 4x4x4 part uses columns 0-3 of B.
    # So B index is (J*2+j)*9 + (K*2+k).
    
    # Let's define the block mappings for the 7 Strassen products.
    # Each product is defined by a set of (I,J) for A and (J,K) for B.
    # And the resulting C block is (I,K).
    
    # M1: A blocks {00, 11}, B blocks {00, 11}. C blocks {00, 11}.
    # But Strassen M1 is a single product of sums.
    # The "A" part is a sum of A00 and A11.
    # The "B" part is a sum of B00 and B11.
    # The result is added to C00 and C11.
    
    # So for M1, we use the 2x2x2 decomposition.
    # The "A" matrix in the 2x2x2 decomposition is the sum of A00 and A11.
    # The "B" matrix is the sum of B00 and B11.
    # The "C" matrix is the sum of C00 and C11.
    
    # This is tricky to map directly to the rank-1 terms because the rank-1 terms are for the *scalar* products.
    # The 2x2x2 decomposition provides 7 rank-1 terms for the product of two 2x2 matrices.
    # So for M1, we generate 7 terms.
    # For each of these 7 terms, the U coefficients come from the A part, V from B part, W from C part.
    
    # Let's iterate through the 7 Strassen products and generate their 7 sub-terms.
    
    # Define the block contributions for each Strassen product M_k
    # M_k is defined by:
    # A_blocks: list of (I, J, coeff)
    # B_blocks: list of (J, K, coeff)
    # C_blocks: list of (I, K, coeff)
    
    # Note: The 2x2x2 decomposition assumes the inputs are 2x2 matrices.
    # Here, the "2x2 matrix" for A is actually a linear combination of 2x2 blocks.
    # So if A_part = A00 + A11, then the local A_{00} corresponds to global A00, local A_{11} corresponds to global A11.
    # Wait, no. The 2x2x2 decomposition works on the entries.
    # If we say "Multiply Matrix X by Matrix Y", and X = A00 + A11, then the entries of X are sums of entries of A00 and A11.
    # So local X_{00} = A00_{00} + A11_{00}.
    # This means the U coefficients for local index 0 will be applied to both global A00 and A11.
    
    # So, for each Strassen product M_k:
    # 1. Identify the A blocks and B blocks involved.
    # 2. For each of the 7 sub-terms in the 2x2x2 decomposition:
    #    - For each local U index (i,j) with coeff c_u:
    #       - For each A block (I,J) with coeff c_block:
    #         - Add to U[global_idx(I,J,i,j)] += c_u * c_block
    #    - Similarly for V and W.
    
    # Let's define the block sets for M1..M7.
    
    # M1: A = A00 + A11, B = B00 + B11, C = C00 + C11
    # A_blocks: [(0,0,1), (1,1,1)]
    # B_blocks: [(0,0,1), (1,1,1)]
    # C_blocks: [(0,0,1), (1,1,1)]
    
    # M2: A = A10 + A11, B = B00, C = C10 + C11
    # A_blocks: [(1,0,1), (1,1,1)]
    # B_blocks: [(0,0,1)]
    # C_blocks: [(1,0,1), (1,1,1)]
    
    # M3: A = A00, B = B01 - B11, C = C01 + C11
    # A_blocks: [(0,0,1)]
    # B_blocks: [(0,1,1), (1,1,-1)]
    # C_blocks: [(0,1,1), (1,1,1)]
    
    # M4: A = A11, B = B10 - B00, C = C00 + C10
    # A_blocks: [(1,1,1)]
    # B_blocks: [(1,0,1), (0,0,-1)]
    # C_blocks: [(0,0,1), (1,0,1)]
    
    # M5: A = A00 + A01, B = B11, C = -C00 + C01
    # A_blocks: [(0,0,1), (0,1,1)]
    # B_blocks: [(1,1,1)]
    # C_blocks: [(0,0,-1), (0,1,1)]
    
    # M6: A = A10 - A00, B = B00 + B01, C = C11
    # A_blocks: [(1,0,1), (0,0,-1)]
    # B_blocks: [(0,0,1), (0,1,1)]
    # C_blocks: [(1,1,1)]
    
    # M7: A = A01 - A11, B = B10 + B11, C = C00
    # A_blocks: [(0,1,1), (1,1,-1)]
    # B_blocks: [(1,0,1), (1,1,1)]
    # C_blocks: [(0,0,1)]
    
    strassen_products = [
        { 'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)] },
        { 'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1), (1,1,1)] },
        { 'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'C': [(0,1,1), (1,1,1)] },
        { 'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'C': [(0,0,1), (1,0,1)] },
        { 'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1), (0,1,1)] },
        { 'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)] },
        { 'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1)] },
    ]
    
    def add_2x2_block(prod_def, offset_rank):
        nonlocal current_rank
        for sub_term in strassen_2x2_terms:
            u_local, v_local, w_local = sub_term
            r = offset_rank
            for (lu, cu) in u_local:
                for (I, J, cA) in prod_def['A']:
                    global_u = (I*2 + lu//2)*4 + (J*2 + lu%2)
                    U[global_u][r] += cu * cA
            for (lv, cv) in v_local:
                for (J, K, cB) in prod_def['B']:
                    global_v = (J*2 + lv//2)*9 + (K*2 + lv%2)
                    V[global_v][r] += cv * cB
            for (lw, cw) in w_local:
                for (I, K, cC) in prod_def['C']:
                    global_w = (K*2 + lw//2)*4 + (I*2 + lw%2)
                    W[global_w][r] += cw * cC
            current_rank += 1
            
    # Add the 7 Strassen products for 4x4x4
    for prod in strassen_products:
        add_2x2_block(prod, current_rank)
        
    # current_rank is now 49.
    
    # --- Part 2: 4x4 * 4x5 (cols 4-8 of B) ---
    # Split into 4x4 (cols 4-7) and 4x1 (col 8).
    
    # Part 2a: 4x4 * 4x4 (cols 4-7)
    # This is identical to Part 1, but B indices are shifted by 4.
    # B index: (J*2+j)*9 + (K*2+k+4)
    # We can reuse the logic by modifying the B index calculation.
    
    # Let's create a modified version of add_2x2_block that takes a B_offset.
    
    def add_2x2_block_offset(prod_def, offset_rank, b_offset):
        nonlocal current_rank
        for sub_term in strassen_2x2_terms:
            u_local, v_local, w_local = sub_term
            r = offset_rank
            for (lu, cu) in u_local:
                for (I, J, cA) in prod_def['A']:
                    global_u = (I*2 + lu//2)*4 + (J*2 + lu%2)
                    U[global_u][r] += cu * cA
            for (lv, cv) in v_local:
                for (J, K, cB) in prod_def['B']:
                    global_v = (J*2 + lv//2)*9 + (K*2 + lv%2 + b_offset)
                    V[global_v][r] += cv * cB
            for (lw, cw) in w_local:
                for (I, K, cC) in prod_def['C']:
                    global_w = (K*2 + lw//2)*4 + (I*2 + lw%2 + b_offset) # W index k*n+i. k is shifted.
                    W[global_w][r] += cw * cC
            current_rank += 1
            
    for prod in strassen_products:
        add_2x2_block_offset(prod, current_rank, 4)
        
    # current_rank is now 49 + 49 = 98.
    
    # Part 2b: 4x4 * 4x1 (col 8)
    # Schoolbook. 16 terms.
    # A is 4x4. B is 4x1 (col 8). C is 4x1 (col 8).
    # U[i*4+j] = 1
    # V[j*9+8] = 1
    # W[8*4+i] = 1
    
    for i in range(4):
        for j in range(4):
            r = current_rank
            U[i*4 + j][r] = 1
            V[j*9 + 8][r] = 1
            W[8*4 + i][r] = 1
            current_rank += 1
            
    # current_rank is now 98 + 16 = 114.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
