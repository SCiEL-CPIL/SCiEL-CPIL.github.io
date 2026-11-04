"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen decomposition at 2x2 block level (3x3 blocks)
    # Each block multiplication uses schoolbook (rank 9 per 3x3, but we need rank 27 for the full 3x3x3 tensor)
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    # Block partition: 6x6 into 2x2 blocks of 3x3
    # A[i,j]: i=0..5, j=0..5
    # A00: i=0..2, j=0..2
    # A01: i=0..2, j=3..5
    # A10: i=3..5, j=0..2
    # A11: i=3..5, j=3..5
    
    # B[j,k]: j=0..5, k=0..5
    # B00: j=0..2, k=0..2
    # B01: j=0..2, k=3..5
    # B10: j=3..5, k=0..2
    # B11: j=3..5, k=3..5
    
    # C[i,k]: i=0..5, k=0..5
    # C00: i=0..2, k=0..2
    # C01: i=0..2, k=3..5
    # C10: i=3..5, k=0..2
    # C11: i=3..5, k=3..5
    
    # W convention: W[k*n+i, r] = W[k*6+i, r] for C[i,k]
    
    # Define the 7 Strassen multiplications
    # Each entry: (A_block_contribs, B_block_contribs, C_block_contribs)
    # A_block_contribs: dict from block_id to coefficient
    # B_block_contribs: dict from block_id to coefficient  
    # C_block_contribs: dict from block_id to coefficient
    
    # Block IDs: 0=A00, 1=A01, 2=A10, 3=A11
    # For B: 0=B00, 1=B01, 2=B10, 3=B11
    # For C: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # For each Strassen term and each schoolbook term in the 3x3 multiplication
    col = 0
    for a_contribs, b_contribs, c_contribs in strassen_terms:
        # a_contribs: {block_id: coeff} for the first factor (P)
        # b_contribs: {block_id: coeff} for the second factor (Q)
        # c_contribs: {block_id: coeff} for the output block(s)
        
        # For the 3x3 schoolbook: P[a,b] * Q[b,c] -> R[a,c]
        # a,b,c in 0..2 (local indices within the 3x3 block)
        
        for la in range(3):  # local row of P (and R)
            for lb in range(3):  # local col of P, local row of Q
                for lc in range(3):  # local col of Q (and R)
                    # P[la, lb] comes from A blocks in a_contribs
                    # Q[lb, lc] comes from B blocks in b_contribs
                    # R[la, lc] goes to C blocks in c_contribs
                    
                    # For each A block that contributes to P:
                    # P[la, lb] = sum over A_blocks of coeff * A_block[la, lb]
                    # A_block[la, lb] corresponds to A[i, j] where:
                    #   i = block_row_offset + la
                    #   j = block_col_offset + lb
                    
                    for a_block, a_coeff in a_contribs.items():
                        a_row_off = (a_block // 2) * 3  # 0 for blocks 0,1; 3 for blocks 2,3
                        a_col_off = (a_block % 2) * 3    # 0 for blocks 0,2; 3 for blocks 1,3
                        gi = a_row_off + la  # global row in A
                        gj = a_col_off + lb  # global col in A
                        a_idx = gi * m + gj  # index in U: i*m + j
                        
                        U[a_idx][col] += a_coeff
                    
                    # For each B block that contributes to Q:
                    for b_block, b_coeff in b_contribs.items():
                        b_row_off = (b_block // 2) * 3  # 0 for blocks 0,1; 3 for blocks 2,3
                        b_col_off = (b_block % 2) * 3    # 0 for blocks 0,2; 3 for blocks 1,3
                        gj2 = b_row_off + lb  # global row in B (= col of A in the product)
                        gk = b_col_off + lc   # global col in B
                        b_idx = gj2 * p + gk  # index in V: j*p + k
                        
                        V[b_idx][col] += b_coeff
                    
                    # For each C block that receives R:
                    for c_block, c_coeff in c_contribs.items():
                        c_row_off = (c_block // 2) * 3
                        c_col_off = (c_block % 2) * 3
                        gi2 = c_row_off + la  # global row in C
                        gk2 = c_col_off + lc  # global col in C
                        # W convention: W[k*n+i, r] so index = gk2 * n + gi2
                        w_idx = gk2 * n + gi2
                        
                        W[w_idx][col] += c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
