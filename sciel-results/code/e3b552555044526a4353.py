import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 28 decomposition for M_{4,4,4} using Strassen
    # U1, V1, W1
    rank_444 = 28
    U1 = np.zeros((16, rank_444))
    V1 = np.zeros((16, rank_444))
    W1 = np.zeros((16, rank_444))
    
    # Strassen terms for 2x2 blocks
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Schoolbook terms for a 2x2 multiplication
    # Indexing: A[i*2+j], B[j*2+k], C[k*2+i]
    # Term 0: A00*B00 -> C00
    # Term 1: A01*B10 -> C00
    # Term 2: A10*B01 -> C10
    # Term 3: A11*B11 -> C11
    
    def add_term(U, V, W, a_idx, b_idx, c_idx, val=1):
        # a_idx: 0..3 (A block index)
        # b_idx: 0..3 (B block index)
        # c_idx: 0..3 (C block index)
        # Within the 2x2 block, we need to add 4 terms
        # But here we are building the 4x4 decomposition directly.
        # The "a_idx" refers to the 2x2 block position in the 4x4 matrix.
        # A00 is rows 0-1, cols 0-1.
        # A01 is rows 0-1, cols 2-3.
        # A10 is rows 2-3, cols 0-1.
        # A11 is rows 2-3, cols 2-3.
        
        # We need to map the 2x2 block indices to the 4x4 matrix indices.
        # This function is a placeholder, I will implement the logic directly.
        pass

    # Let's construct the 28 terms explicitly.
    # Each Strassen term is a 2x2 matrix multiplication.
    # We iterate through the 7 Strassen terms.
    
    # Strassen Coefficients:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # For each M_s, we have a linear combination of A blocks and B blocks.
    # Let A_comb be the 2x2 matrix of coefficients for A blocks.
    # Let B_comb be the 2x2 matrix of coefficients for B blocks.
    # The product is sum_{i,j} A_comb[i,j] * B_comb[i,j] * (A_block_ij * B_block_ij)
    # Wait, the product is (A_comb * A_blocks) * (B_comb * B_blocks).
    # This is a matrix multiplication of two 2x2 matrices of blocks.
    # We can decompose this into 4 scalar (block) multiplications? No, it's a 2x2 matrix multiplication.
    # So for each Strassen term s, we perform a 2x2 multiplication of the combined blocks.
    # This 2x2 multiplication has rank 4.
    
    # So for each s in 0..6:
    #   We define A_s and B_s as 2x2 matrices of linear combinations of the original A and B blocks.
    #   Then we compute A_s * B_s using schoolbook (4 terms).
    #   Each term in A_s * B_s contributes to the output C blocks.
    
    # Let's define the A_s and B_s coefficients.
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    
    strassen_defs = [
        # (A_coeffs, B_coeffs)
        # A_coeffs is 2x2, B_coeffs is 2x2
        # M0
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]]), # (A00+A11)(B00+B11)
        # M1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]]), # (A10+A11)B00
        # M2
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]]), # A00(B01-B11)
        # M3
        ([[0, 0], [0, 1]], [[0, 0], [1, -1]]), # A11(B10-B00)
        # M4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]]), # (A00+A01)B11
        # M5
        ([[ -1, 0], [1, 0]], [[1, 1], [0, 0]]), # (A10-A00)(B00+B01)
        # M6
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]]), # (A01-A11)(B10+B11)
    ]
    
    # Output C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    c_combinations = [
        # For each C block, list of (strassen_term_idx, coeff)
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    # We need to fill U1, V1, W1.
    # U1 shape (16, 28).
    # V1 shape (16, 28).
    # W1 shape (16, 28).
    
    term_col = 0
    for s in range(7):
        A_coeff, B_coeff = strassen_defs[s]
        # A_coeff is 2x2, indices [i,j]
        # B_coeff is 2x2, indices [i,j]
        
        # The product is sum_{i,j} A_coeff[i,j] * B_coeff[i,j] * (A_block_{i,j} * B_block_{i,j})
        # Wait, the matrix product is:
        # (A_comb * B_comb)_{kl} = sum_{i} A_comb[k,i] B_comb[i,l]
        # Here A_comb and B_comb are 2x2 matrices of block coefficients.
        # So we have 4 terms in the 2x2 product.
        for k in range(2): # Output row block
            for l in range(2): # Output col block
                # Term index in the 2x2 product: 0,1,2,3
                # Corresponds to A_block_{k,i} * B_block_{i,l} for i=0,1?
                # No, the sum is over i.
                # So for fixed k,l, we have 2 terms (i=0, i=1).
                # Total 4*2 = 8 terms? No, 2x2 multiplication has 4 output elements, each is a sum of 2 products.
                # So 8 scalar (block) products?
                # No, the rank of 2x2 multiplication is 4.
                # The 4 terms are:
                # 1. A00 B00 -> C00
                # 2. A01 B10 -> C00
                # 3. A10 B01 -> C10
                # 4. A11 B11 -> C11
                # This is for a generic 2x2 matrix.
                
                # In our case, the "matrices" are linear combinations.
                # So we just expand the product.
                # Product = sum_{i=0}^1 A_comb[k,i] B_comb[i,l] * (A_block_{k,i} * B_block_{i,l})
                # This is not quite right. A_comb[k,i] is a coefficient for A_block_{k,i}.
                # So the term is coeff_A * A_block_{k,i} * coeff_B * B_block_{i,l}.
                # This is a single block multiplication.
                
                # So for each k,l, we sum over i=0,1.
                # That gives 2 terms per output block.
                # Total 4 output blocks * 2 terms = 8 terms?
                # But rank is 4. Why?
                # Because the 4 terms in schoolbook are independent.
                # Here, the "A" and "B" are themselves sums.
                # So we are just adding linear combinations.
                
                # Let's just iterate over the 4 schoolbook terms of the 2x2 multiplication.
                # Term 0: A00 B00
                # Term 1: A01 B10
                # Term 2: A10 B01
                # Term 3: A11 B11
                
                # For each of these 4 terms, we have a specific A_block and B_block.
                # A_block index: 0->00, 1->01, 2->10, 3->11.
                # B_block index: 0->00, 1->01, 2->10, 3->11.
                
                # Term 0: A00 (idx 0), B00 (idx 0)
                # Term 1: A01 (idx 1), B10 (idx 2)
                # Term 2: A10 (idx 2), B01 (idx 1)
                # Term 3: A11 (idx 3), B11 (idx 3)
                
                schoolbook_terms = [
                    (0, 0), # A00, B00
                    (1, 2), # A01, B10
                    (2, 1), # A10, B01
                    (3, 3)  # A11, B11
                ]
                
                for tb in range(4):
                    a_block_idx, b_block_idx = schoolbook_terms[tb]
                    # a_block_idx: 0,1,2,3
                    # b_block_idx: 0,1,2,3
                    
                    # We need the coefficient of this block in A_comb and B_comb.
                    # A_comb is 2x2. A_block_idx maps to [row, col].
                    # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    ar, ac = divmod(a_block_idx, 2)
                    br, bc = divmod(b_block_idx, 2)
                    
                    coeff_a = A_coeff[ar][ac]
                    coeff_b = B_coeff[br][bc]
                    
                    if coeff_a == 0 or coeff_b == 0:
                        continue
                    
                    coeff = coeff_a * coeff_b
                    
                    # Now, this term contributes to all C blocks that depend on M_s.
                    # And specifically, which C block does this schoolbook term produce?
                    # Schoolbook term 0 (A00 B00) produces C00 (idx 0).
                    # Term 1 (A01 B10) produces C00 (idx 0).
                    # Term 2 (A10 B01) produces C10 (idx 2).
                    # Term 3 (A11 B11) produces C11 (idx 3).
                    
                    c_block_idx = [0, 0, 2, 3][tb]
                    
                    # Now, for each C block that uses M_s, we add this term.
                    for c_idx, c_coeff in c_combinations[c_block_idx].items():
                        if c_idx == s:
                            final_coeff = coeff * c_coeff
                            if final_coeff != 0:
                                # Add to U1, V1, W1
                                # U1: A_block_idx
                                # V1: B_block_idx
                                # W1: C_block_idx (which is c_block_idx, not c_idx)
                                # Wait, c_block_idx is the output of the 2x2 multiplication.
                                # The final output C_block is determined by c_block_idx.
                                # The coefficient is final_coeff.
                                
                                # U1 row: A_block_idx
                                # V1 row: B_block_idx
                                # W1 row: c_block_idx
                                
                                U1[a_block_idx, term_col] = final_coeff
                                V1[b_block_idx, term_col] = 1 # Wait, V is for B.
                                # The term is A_block * B_block.
                                # So U gets A_block, V gets B_block.
                                # But we have coeff_a and coeff_b.
                                # We can put coeff_a in U and coeff_b in V.
                                # Or coeff in U and 1 in V.
                                # Let's put coeff_a in U and coeff_b in V.
                                
                                U1[a_block_idx, term_col] = coeff_a
                                V1[b_block_idx, term_col] = coeff_b
                                W1[c_block_idx, term_col] = c_coeff
                                
                                term_col += 1
                                
    # This logic is flawed because term_col increments inside the c_combinations loop,
    # but we only want one column per (s, tb) pair if it's non-zero.
    # And the U, V, W entries should be set for that column.
    
    # Let's restart the construction of U1, V1, W1 more carefully.
    U1 = np.zeros((16, 28))
    V1 = np.zeros((16, 28))
    W1 = np.zeros((16, 28))
    
    col = 0
    for s in range(7):
        A_coeff, B_coeff = strassen_defs[s]
        for tb in range(4):
            a_block_idx, b_block_idx = schoolbook_terms[tb]
            ar, ac = divmod(a_block_idx, 2)
            br, bc = divmod(b_block_idx, 2)
            
            coeff_a = A_coeff[ar][ac]
            coeff_b = B_coeff[br][bc]
            
            if coeff_a == 0 or coeff_b == 0:
                continue
            
            c_block_idx = [0, 0, 2, 3][tb]
            
            # This term contributes to C_block_idx with coefficient c_coeff
            # But c_coeff depends on which C block we are computing.
            # Wait, the schoolbook term produces a specific C_block.
            # That C_block is then combined with others to form the final C.
            # So we just add this term to the final C_block.
            
            # The final C_block index is c_block_idx.
            # The coefficient is coeff_a * coeff_b.
            
            # But we must check if this M_s term is used in the final C_block.
            # Yes, all M_s terms are used in some C blocks.
            # But for a specific c_block_idx, only some M_s terms contribute.
            # We need to add the term to the W matrix for the c_block_idx.
            
            # Wait, the W matrix has rows for each output element.
            # The output element is (k, i) where k is the block row and i is the element in the block.
            # Here c_block_idx is the block index.
            # So we need to add to W1[c_block_idx, col].
            
            # But what is the coefficient?
            # It is coeff_a * coeff_b.
            
            # However, we must ensure that this term is actually part of the final sum for that C_block.
            # The c_combinations dict tells us which M_s terms contribute to which C_block.
            # So we check if s is in c_combinations[c_block_idx].
            
            if s in c_combinations[c_block_idx]:
                c_coeff = c_combinations[c_block_idx][s]
                if c_coeff != 0:
                    U1[a_block_idx, col] = coeff_a
                    V1[b_block_idx, col] = coeff_b
                    W1[c_block_idx, col] = c_coeff
                    col += 1
    # col should be 28.
    
    # Now construct U2, V2, W2 for M_{4,4,3}
    U2 = U1
    V2 = V1
    W2 = W1[:12, :] # First 12 rows (k=0,1,2)
    
    # Concatenate
    U_total = np.hstack([U1, U2])
    V_total = np.hstack([V1, V2])
    W_total = np.vstack([W1, W2])
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
