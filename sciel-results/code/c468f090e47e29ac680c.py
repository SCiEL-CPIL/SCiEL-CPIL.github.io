"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 Strassen-based decomposition for 4x4 matrix multiplication.
    # A (4x4) and B (4x4) are split into 2x2 blocks of 2x2 matrices.
    # Strassen computes 7 products of 2x2 matrices, each via Strassen on 1x1 scalars.
    # Total: 7 * 7 = 49 scalar multiplications.
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Block structure: A is split into 4 blocks A00, A01, A10, A11 (each 2x2)
    # Index mapping: row i, col j -> index i*m + j = i*4 + j
    # Block A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block A01: rows 0-1, cols 2-3 -> indices 0,1,6,7
    # Block A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block A11: rows 2-3, cols 2-3 -> indices 8,9,14,15
    
    # Similarly for B: B00, B01, B10, B11
    # Block B00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block B01: rows 0-1, cols 2-3 -> indices 0,1,6,7
    # Block B10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block B11: rows 2-3, cols 2-3 -> indices 8,9,14,15
    
    # Output C blocks: C00, C01, C10, C11
    # C00 = M1 + M4 - M5 - M7
    # C01 = M2 + M3
    # C10 = M6 + M7
    # C11 = M1 - M3 + M4 + M5
    
    # Where:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each M_r is a 2x2 matrix product, decomposed via Strassen into 7 scalar multiplications.
    # For a 2x2 matrix product P = Q * R, where Q and R are 2x2:
    # P00 = S1 + S4 - S5 - S7
    # P01 = S2 + S3
    # P10 = S6 + S7
    # P11 = S1 - S3 + S4 + S5
    # Where:
    # S1 = (Q00 + Q11)(R00 + R11)
    # S2 = (Q10 + Q11)R00
    # S3 = Q00(R01 - R11)
    # S4 = Q11(R10 - R00)
    # S5 = (Q00 + Q01)R11
    # S6 = (Q10 - Q00)(R00 + R01)
    # S7 = (Q01 - Q11)(R10 + R11)
    
    # We need to express each of the 49 scalar multiplications as triples (u_row, v_row, w_row) with coefficients.
    
    # Let's define helper to get the indices for a 2x2 block.
    # For A block at (bi, bj) where bi, bj in {0,1}:
    #   The 4 entries are at positions:
    #   (bi*2, bj*2), (bi*2, bj*2+1), (bi*2+1, bj*2), (bi*2+1, bj*2+1)
    #   Which map to linear indices: (bi*2)*4 + (bj*2) = bi*8 + bj*2, etc.
    
    # Similarly for B and W.
    
    # Let's build the 49 triples.
    
    # For each of the 7 Strassen steps at the top level (r1 in 0..6),
    # and for each of the 7 Strassen steps at the bottom level (r2 in 0..6),
    # we have one scalar multiplication.
    
    # The top-level Strassen defines linear combinations of A-blocks and B-blocks.
    # The bottom-level Strassen defines linear combinations of scalar entries.
    
    # Let's define the top-level coefficients.
    # For each top-level step r1, we have coefficients for A-blocks (a00, a01, a10, a11) and B-blocks (b00, b01, b10, b11).
    
    top_level = [
        # (A_coeffs: [a00, a01, a10, a11], B_coeffs: [b00, b01, b10, b11])
        ([1, 0, 0, 1], [1, 0, 0, 1]),  # M1 = (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0]),  # M2 = (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3 = A00(B01-B11)
        ([0, 0, 0, 1], [0, 0, 1, -1]), # M4 = A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1]),  # M5 = (A00+A01)B11
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6 = (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1]), # M7 = (A01-A11)(B10+B11)
    ]
    
    # Bottom-level Strassen coefficients for a 2x2 matrix product.
    # For Q (2x2) and R (2x2), the 7 products:
    # Q entries: q00, q01, q10, q11
    # R entries: r00, r01, r10, r11
    bottom_level = [
        # (Q_coeffs: [q00, q01, q10, q11], R_coeffs: [r00, r01, r10, r11])
        ([1, 0, 0, 1], [1, 0, 0, 1]),  # S1
        ([0, 0, 1, 1], [1, 0, 0, 0]),  # S2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # S3
        ([0, 0, 0, 1], [0, 0, 1, -1]), # S4
        ([1, 1, 0, 0], [0, 0, 0, 1]),  # S5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # S6
        ([0, 1, 0, -1], [0, 0, 1, 1]), # S7
    ]
    
    # Now, for each top-level step r1 and bottom-level step r2, we determine the scalar entries involved.
    
    # A block (bi, bj) has scalar entries at:
    #   (bi*2, bj*2), (bi*2, bj*2+1), (bi*2+1, bj*2), (bi*2+1, bj*2+1)
    # Linear index in A: (bi*2)*4 + (bj*2) = bi*8 + bj*2
    #   Entry (0,0) of block: index bi*8 + bj*2
    #   Entry (0,1) of block: index bi*8 + bj*2 + 1
    #   Entry (1,0) of block: index bi*8 + bj*2 + 4
    #   Entry (1,1) of block: index bi*8 + bj*2 + 5
    
    # B block similarly.
    
    # For the output, the block C(bi, bj) gets contributions.
    # C block (bi, bj) entries:
    #   (bi*2, bj*2), (bi*2, bj*2+1), (bi*2+1, bj*2), (bi*2+1, bj*2+1)
    # Linear index in W (which is C transposed in a sense, but W[k*n+i]):
    #   W index for C(i,k) is k*4 + i
    #   For C block (bi, bj), entry (di, dj) where di, dj in {0,1}:
    #     i = bi*2 + di, k = bj*2 + dj
    #     W index = (bj*2 + dj)*4 + (bi*2 + di)
    
    # The top-level Strassen gives linear combinations of C-blocks.
    # M1 contributes to C00 (+1), C11 (+1)
    # M2 contributes to C10 (+1)
    # M3 contributes to C01 (+1)
    # M4 contributes to C11 (+1)
    # M5 contributes to C00 (-1), C11 (+1)
    # M6 contributes to C10 (+1)
    # M7 contributes to C01 (+1), C11 (-1)
    
    # More precisely:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M2 + M3
    # C10 = M6 + M7
    # C11 = M1 - M3 + M4 + M5
    
    top_level_output = [
        # (C00, C01, C10, C11) coefficients for each M_r
        ([1, 0, 0, 0]),   # M1: not quite, M1 contributes to C00 and C11
    ]
    
    # Let me redo this. The top-level Strassen says:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M2 + M3
    # C10 = M6 + M7
    # C11 = M1 - M3 + M4 + M5
    
    # So for each M_r, its contribution to each C-block:
    top_level_output = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, 0],   # M2 -> C10
        [0, 1, 0, -1],  # M3 -> C01, C11
        [0, 0, 0, 1],   # M4 -> C11
        [-1, 0, 0, 1],  # M5 -> C00, C11
        [0, 0, 1, 0],   # M6 -> C10
        [0, 1, 0, -1],  # M7 -> C01, C11
    ]
    
    # Now build the 49 triples.
    column = 0
    for r1 in range(7):
        A_block_coeffs = top_level[r1][0]  # [a00, a01, a10, a11]
        B_block_coeffs = top_level[r1][1]  # [b00, b01, b10, b11]
        
        for r2 in range(7):
            Q_coeffs = bottom_level[r2][0]  # [q00, q01, q10, q11]
            R_coeffs = bottom_level[r2][1]  # [r00, r01, r10, r11]
            
            # The Q matrix is a linear combination of A-blocks.
            # Q is 2x2, with entries q00, q01, q10, q11.
            # Each q entry is a linear combination of scalar entries from A-blocks.
            
            # Similarly for R (B-blocks).
            
            # For U (A-side):
            # We need to determine which scalar entries of A are involved and with what coefficients.
            # U row index corresponds to A entry (i, j) -> index i*4 + j.
            
            # Q is formed as: Q = sum over blocks of A_block_coeff * A_block
            # But Q is 2x2, and each A_block is 2x2.
            # So Q00 = sum_{bi,bj} A_block_coeffs[bi*2+bj] * A_block(bi,bj)[0,0]
            # etc.
            
            # Actually, the top-level Strassen combines entire 2x2 blocks.
            # M1 = (A00 + A11)(B00 + B11)
            # So Q = A00 + A11, which is a 2x2 matrix.
            # Q[0,0] = A00[0,0] + A11[0,0]
            # Q[0,1] = A00[0,1] + A11[0,1]
            # Q[1,0] = A00[1,0] + A11[1,0]
            # Q[1,1] = A00[1,1] + A11[1,1]
            
            # So for Q_coeffs = [q00, q01, q10, q11], the coefficient for each scalar entry is:
            # For A entry at block (bi,bj), position (di,dj):
            #   The coefficient is A_block_coeffs[bi*2+bj] * Q_coeffs[di*2+dj]
            
            # Similarly for B.
            
            # For W (output side):
            # The output C-block coefficients from top_level_output[r1] give the contribution to each C-block.
            # The bottom-level Strassen output gives the position within the C-block.
            
            # Bottom-level Strassen output:
            # P00 = S1 + S4 - S5 - S7
            # P01 = S2 + S3
            # P10 = S6 + S7
            # P11 = S1 - S3 + S4 + S5
            
            bottom_level_output = [
                [1, 0, 0, 1],   # S1 -> P00, P11
                [0, 1, 0, 0],   # S2 -> P01
                [0, 1, 0, -1],  # S3 -> P01, P11
                [0, 0, 0, 1],   # S4 -> P11
                [-1, 0, 0, 1],  # S5 -> P00, P11
                [0, 0, 1, 0],   # S6 -> P10
                [0, 1, 0, -1],  # S7 -> P01, P11
            ]
            
            # Now, for this particular (r1, r2), we have:
            # - A-side: linear combination of A scalar entries
            # - B-side: linear combination of B scalar entries  
            # - C-side: linear combination of C scalar entries
            
            # The scalar multiplication is: (linear_comb_A) * (linear_comb_B) contributes to (linear_comb_C)
            
            # For U: for each A scalar entry, the coefficient is the product of A_block_coeff and Q_coeff.
            # But U is a matrix where each row corresponds to one A entry, and each column to one term.
            # So U[entry_index, column] = coefficient of that A entry in the linear combination.
            
            # Let's collect all A entries involved.
            A_entries = {}  # entry_index -> coefficient
            for bi in range(2):
                for bj in range(2):
                    block_coeff = A_block_coeffs[bi * 2 + bj]
                    if block_coeff == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            q_coeff = Q_coeffs[di * 2 + dj]
                            if q_coeff == 0:
                                continue
                            a_entry_index = (bi * 2 + di) * 4 + (bj * 2 + dj)
                            coeff = block_coeff * q_coeff
                            A_entries[a_entry_index] = A_entries.get(a_entry_index, 0) + coeff
            
            B_entries = {}
            for bi in range(2):
                for bj in range(2):
                    block_coeff = B_block_coeffs[bi * 2 + bj]
                    if block_coeff == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            r_coeff = R_coeffs[di * 2 + dj]
                            if r_coeff == 0:
                                continue
                            b_entry_index = (bi * 2 + di) * 4 + (bj * 2 + dj)
                            coeff = block_coeff * r_coeff
                            B_entries[b_entry_index] = B_entries.get(b_entry_index, 0) + coeff
            
            C_entries = {}
            top_out = top_level_output[r1]  # [C00, C01, C10, C11]
            bot_out = bottom_level_output[r2]  # [P00, P01, P10, P11]
            for bi in range(2):
                for bj in range(2):
                    c_block_coeff = top_out[bi * 2 + bj]
                    if c_block_coeff == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            p_coeff = bot_out[di * 2 + dj]
                            if p_coeff == 0:
                                continue
                            # C entry at (i, k) where i = bi*2 + di, k = bj*2 + dj
                            # W index = k*4 + i
                            c_entry_index = (bj * 2 + dj) * 4 + (bi * 2 + di)
                            coeff = c_block_coeff * p_coeff
                            C_entries[c_entry_index] = C_entries.get(c_entry_index, 0) + coeff
            
            # Now set U, V, W for this column.
            for entry_idx, coeff in A_entries.items():
                U[entry_idx][column] = coeff
            for entry_idx, coeff in B_entries.items():
                V[entry_idx][column] = coeff
            for entry_idx, coeff in C_entries.items():
                W[entry_idx][column] = coeff
            
            column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
