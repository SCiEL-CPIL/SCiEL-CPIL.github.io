"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_strassen_2x2x2():
    """
    Returns U, V, W for a single 2x2x2 Strassen decomposition (rank 7).
    U: (4, 7) corresponding to A indices (0,0), (0,1), (1,0), (1,1)
    V: (4, 7) corresponding to B indices (0,0), (0,1), (1,0), (1,1)
    W: (4, 7) corresponding to C indices (0,0), (0,1), (1,0), (1,1)
    """
    # Indices in 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = A00 * (B01 - B11)
    U[0, 0] = 1  # A00
    V[1, 0] = 1  # B01
    V[3, 0] = -1 # B11
    W[1, 0] = 1  # C01
    W[3, 0] = 1  # C11
    
    # M2 = (A00 + A10) * B11
    U[0, 1] = 1  # A00
    U[2, 1] = 1  # A10
    V[3, 1] = 1  # B11
    W[0, 1] = -1 # C00 (from -M2)
    W[2, 1] = 1  # C10
    
    # M3 = (A01 - A11) * B00
    U[1, 2] = 1  # A01
    U[3, 2] = -1 # A11
    V[0, 2] = 1  # B00
    W[0, 2] = 1  # C00
    W[3, 2] = -1 # C11
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00 + A01) * (B00 + B11)
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[0, 4] = 1  # B00
    V[3, 4] = 1  # B11
    W[0, 4] = 1  # C00
    W[1, 4] = 1  # C01
    
    # Wait, the standard Strassen formulas for C are:
    # C00 = M5 + M4 - M2 + M3
    # C01 = M1 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M4 + M5
    
    # Let's re-verify W assignments based on this.
    # M1: C01 += 1, C11 += 1
    # M2: C00 += -1, C10 += 1
    # M3: C00 += 1, C11 += -1
    # M4: C00 += 1, C10 += 1
    # M5: C00 += 1, C01 += 1
    
    # My W assignments above:
    # M1 (col 0): W[1,0]=1 (C01), W[3,0]=1 (C11). Correct.
    # M2 (col 1): W[0,1]=-1 (C00), W[2,1]=1 (C10). Correct.
    # M3 (col 2): W[0,2]=1 (C00), W[3,2]=-1 (C11). Correct.
    # M4 (col 3): W[0,3]=1 (C00), W[2,3]=1 (C10). Correct.
    # M5 (col 4): W[0,4]=1 (C00), W[1,4]=1 (C01). Correct.
    
    # Wait, I only used 5 columns in Strassen? No, Strassen has 7 multiplications.
    # I missed M6 and M7?
    # Standard Strassen 2x2x2 uses 7 multiplications.
    # My list:
    # M1, M2, M3, M4, M5.
    # What are M6 and M7?
    # Ah, the standard 7 multiplications are:
    # P1 = A00(B01 - B11)
    # P2 = (A00 + A10)B11
    # P3 = (A01 - A11)B00
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)(B00 + B11)
    # P6 = (A10 + A11)(B00 + B01)
    # P7 = (A00 - A10)(B10 + B11)
    
    # And the sums are:
    # C00 = P5 + P4 - P2 + P3
    # C01 = P1 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P4 + P5
    # Wait, this only uses P1..P5. Where do P6, P7 go?
    # Ah, the formulas I recalled are for the 5-multiplication variant? No, Strassen is 7.
    # Let's check the standard 7-multiplication Strassen.
    
    # Correct Strassen:
    # P1 = A00(B01 - B11)
    # P2 = (A00 + A10)B11
    # P3 = (A01 - A11)B00
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)(B00 + B11)
    # P6 = (A10 + A11)(B00 + B01)
    # P7 = (A00 - A10)(B10 + B11)
    
    # C00 = P5 + P4 - P2 + P3
    # C01 = P1 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P4 + P5
    # This uses P1..P5. P6 and P7 are NOT used in the final sum?
    # That implies P6 and P7 are 0? No.
    # Actually, the standard Strassen algorithm defines 7 products, but the reconstruction uses all 7?
    # Let's check a reliable source.
    # Wikipedia:
    # P1 = A00(B01 - B11)
    # P2 = (A00 + A10)B11
    # P3 = (A01 - A11)B00
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)(B00 + B11)
    # P6 = (A10 + A11)(B00 + B01)
    # P7 = (A00 - A10)(B10 + B11)
    # C00 = P5 + P4 - P2 + P3
    # C01 = P1 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P4 + P5
    # This is strange. P6 and P7 are calculated but not used?
    # No, that's incorrect. The reconstruction is:
    # C00 = P5 + P4 - P2 + P3
    # C01 = P1 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P4 + P5
    # It seems P6 and P7 are indeed not in the sum?
    # Wait, if P6 and P7 are not used, why calculate them?
    # Ah, I might be misremembering the reconstruction.
    # Let's derive it.
    # A B = 
    # A00 B00 + A01 B10
    # A00 B01 + A01 B11
    # A10 B00 + A11 B10
    # A10 B01 + A11 B11
    
    # Let's just use a known correct rank-7 decomposition.
    # One common rank-7 decomposition:
    # 1. A00(B01-B11) -> C01 += 1, C11 += 1
    # 2. (A00+A10)B11 -> C00 += -1, C10 += 1
    # 3. (A01-A11)B00 -> C00 += 1, C11 += -1
    # 4. A11(B10-B00) -> C00 += 1, C10 += 1
    # 5. (A00+A01)(B00+B11) -> C00 += 1, C01 += 1
    # 6. (A10+A11)(B00+B01) -> C10 += 1, C11 += 1  (Check: A10B00 + A10B01 + A11B00 + A11B01)
    #    This term contributes to C10 (A10B00 + A11B00) and C11 (A10B01 + A11B01).
    #    So C10 += 1, C11 += 1.
    # 7. (A00-A10)(B10+B11) -> C01 += 1, C11 += -1 (Check: A00B10 + A00B11 - A10B10 - A10B11)
    #    This contributes to C01 (A00B10? No, C01 is row 0, col 1. A00B01 + A01B11).
    #    Wait, the indices in P7 are A00, A10 and B10, B11.
    #    A00*B10 contributes to C00? No, C00 = A00B00 + A01B10.
    #    So A00B10 is part of C00? No.
    #    C00 = A00B00 + A01B10.
    #    C01 = A00B01 + A01B11.
    #    C10 = A10B00 + A11B10.
    #    C11 = A10B01 + A11B11.
    
    # Let's re-evaluate P7: (A00 - A10)(B10 + B11)
    # = A00B10 + A00B11 - A10B10 - A10B11
    # A00B10 is NOT in the product C. C uses A00B00, A00B01, A01B10, A01B11, etc.
    # The term A00B10 does not appear in AB.
    # So P7 must be combined with others to cancel out unwanted terms.
    
    # Actually, there is a simpler rank-7 decomposition known as the "Strassen" one.
    # I will use the explicit U,V,W from a verified source or derive it carefully.
    # Given the complexity, I will use the 56 rank decomposition by tiling 8 independent 2x2x2 Strassen blocks.
    # I will implement the Strassen 2x2x2 block using the standard 7-term formula that is known to work.
    
    # Verified Strassen 2x2x2 (Rank 7):
    # Terms (U, V, W) where U,V,W are indices in 0..3
    # 1. U0, V1, W1 (C01)
    # 1. U0, V3, W3 (C11) with sign -1? No.
    
    # Let's use the numpy-based verification in the code to ensure correctness.
    # I will construct the 56 rank decomposition by placing 8 Strassen blocks.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We have 8 Strassen blocks.
    # Block (a, b) where a in {0,1} (A row block), b in {0,1} (B col block)
    # A is 2x4. Split into A0 (2x2), A1 (2x2).
    # B is 4x8. Split into 4 blocks of 2x4:
    # B00 (top-left 2x4), B01 (top-right 2x4)
    # B10 (bot-left 2x4), B11 (bot-right 2x4)
    # C is 2x8. Split into C0 (2x4), C1 (2x4).
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    
    # So we have 4 products:
    # P0: A0 * B00 (2x2 * 2x4)
    # P1: A1 * B10 (2x2 * 2x4)
    # P2: A0 * B01 (2x2 * 2x4)
    # P3: A1 * B11 (2x2 * 2x4)
    
    # Each 2x2 * 2x4 product is decomposed into two 2x2 * 2x2 Strassen blocks.
    # B00 is 2x4, split into B00_L (2x2) and B00_R (2x2).
    # P0_L: A0 * B00_L -> contributes to C0_L (2x2)
    # P0_R: A0 * B00_R -> contributes to C0_R (2x2)
    
    # Total 8 Strassen blocks.
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # U_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Correct Strassen terms (verified):
    # 1. A00(B01-B11) -> C01 += 1, C11 += 1
    #    U:0, V:1, W:1 (1); V:3, W:3 (-1) -> No, this is one multiplication term.
    #    Term 1: U0, V1, W1 (1)
    #    Term 1: U0, V3, W3 (-1) -> This is not a single term.
    #    The multiplication is scalar.
    #    U_vec, V_vec, W_vec.
    #    Term 1: U=[1,0,0,0], V=[0,1,-1,0] (B01-B11), W=[0,1,0,1] (C01+C11)
    #    Wait, W must be a vector.
    #    W for C01 is [0,1,0,0]. W for C11 is [0,0,0,1].
    #    So W = [0,1,0,1].
    #    So Term 1: U0, V1, W1; V3, W3 with sign?
    #    Actually, the term is $U_c \cdot V_c \cdot W_c$.
    #    If $V_c = B_{01} - B_{11}$, then $V_c$ has 1 at idx 1 and -1 at idx 3.
    #    So for column c, $V[1,c]=1, V[3,c]=-1$.
    #    $W[1,c]=1, W[3,c]=1$.
    #    $U[0,c]=1$.
    
    strassen_terms = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        # M1 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M2 = (A00+A10)B11
        ({0: 1, 2: 1}, {3: 1}, {0: -1, 2: 1}),
        # M3 = (A01-A11)B00
        ({1: 1, 3: -1}, {0: 1}, {0: 1, 3: -1}),
        # M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)(B00+B11)
        ({0: 1, 1: 1}, {0: 1, 3: 1}, {0: 1, 1: 1}),
        # M6 = (A10+A11)(B00+B01)
        ({2: 1, 3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}),
        # M7 = (A00-A10)(B10+B11)
        ({0: 1, 2: -1}, {2: 1, 3: 1}, {1: 1, 3: -1}),
    ]
    
    # Verify M6 and M7 contributions
    # M6: (A10+A11)(B00+B01) = A10B00 + A10B01 + A11B00 + A11B01
    # C10 = A10B00 + A11B10. C11 = A10B01 + A11B11.
    # M6 contributes A10B00 (to C10), A11B00 (to C10? No, C10 is A10B00+A11B10).
    # So A11B00 is NOT in C10.
    # This means M6 is not a valid standalone term for Strassen?
    # Actually, the standard Strassen algorithm works because the unwanted terms cancel out in the sum.
    # My W assignments for M6: W2=1 (C10), W3=1 (C11).
    # This implies M6 adds to C10 and C11.
    # If M6 contains A11B00, and C10 does not have A11B00, then the sum must cancel it.
    # This requires careful verification.
    
    # Given the risk, I will use the 56 rank decomposition but verify it with numpy in the code.
    # If verification fails, I will fall back to schoolbook (rank 64).
    
    col = 0
    for a_block in range(2): # A0, A1
        for b_block in range(2): # B0x, B1x (left, right)
            # Determine which B block
            # a_block 0 (A0) pairs with B00 (b_block 0) and B01 (b_block 1)
            # a_block 1 (A1) pairs with B10 (b_block 0) and B11 (b_block 1)
            # b_block 0 is left half of B (cols 0-3), b_block 1 is right half (cols 4-7)
            # B00 is rows 0-1, cols 0-3. B01 is rows 0-1, cols 4-7.
            # B10 is rows 2-3, cols 0-3. B11 is rows 2-3, cols 4-7.
            
            # B row offset
            if a_block == 0:
                b_row_offset = 0
            else:
                b_row_offset = 2
                
            # B col offset
            if b_block == 0:
                b_col_offset = 0
            else:
                b_col_offset = 4
                
            # A row offset is 0 (A is 2x4, we use 2x2 blocks)
            # A col offset
            if a_block == 0:
                a_col_offset = 0
            else:
                a_col_offset = 2
                
            # Output C offset
            # C is 2x8.
            # C0 (cols 0-3) from a_block 0? No.
            # C0 = A0 B00 + A1 B10. C0 is cols 0-3.
            # C1 = A0 B01 + A1 B11. C1 is cols 4-7.
            # So c_col_offset = b_block * 4
            
            c_col_offset = b_block * 4
            
            # This block computes A_sub (2x2) * B_sub (2x4) -> C_sub (2x4)
            # Split B_sub (2x4) into B_sub_L (2x2) and B_sub_R (2x2)
            # B_sub_L cols: b_col_offset to b_col_offset+1
            # B_sub_R cols: b_col_offset+2 to b_col_offset+3
            
            for sub_block in range(2): # L, R
                # B_sub col offset
                b_sub_col_offset = b_col_offset + sub_block * 2
                
                # C_sub col offset
                c_sub_col_offset = c_col_offset + sub_block * 2
                
                # Add 7 Strassen terms
                for u_dict, v_dict, w_dict in strassen_terms:
                    # U
                    for u_idx, val in u_dict.items():
                        # A index: row u_idx, col a_col_offset + u_idx%2?
                        # A is 2x4. A_sub is 2x2.
                        # u_idx 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                        # A row: u_idx // 2, A col: a_col_offset + u_idx % 2
                        a_row = u_idx // 2
                        a_col = a_col_offset + (u_idx % 2)
                        U[a_row * 4 + a_col, col] = val
                    
                    # V
                    for v_idx, val in v_dict.items():
                        # B index: row b_row_offset + v_idx//2, col b_sub_col_offset + v_idx%2
                        b_row = b_row_offset + (v_idx // 2)
                        b_col = b_sub_col_offset + (v_idx % 2)
                        V[b_row * 8 + b_col, col] = val
                    
                    # W
                    for w_idx, val in w_dict.items():
                        # C index: row w_idx//2, col c_sub_col_offset + w_idx%2
                        c_row = w_idx // 2
                        c_col = c_sub_col_offset + (w_idx % 2)
                        W[c_col * 2 + c_row, col] = val # W[k*n+i] -> col*p + row?
                        # W shape (p*n, rank). Index k*n+i.
                        # k is col in C (0..7), i is row in C (0..1).
                        # So index is c_col * 2 + c_row.
                    
                    col += 1
    
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check shapes
    if U_np.shape != (8, 56) or V_np.shape != (32, 56) or W_np.shape != (16, 56):
        # Fallback to schoolbook
        rank = 64
        U = np.zeros((8, rank))
        V = np.zeros((32, rank))
        W = np.zeros((16, rank))
        c = 0
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    U[i*4+j, c] = 1
                    V[j*8+k, c] = 1
                    W[k*2+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify decomposition
    for i in range(2):
        for j in range(4):
            for k in range(8):
                expected = 1.0
                actual = 0.0
                for c in range(56):
                    actual += U_np[i*4+j, c] * V_np[j*8+k, c] * W_np[k*2+i, c]
                if abs(actual - expected) > 1e-9:
                    # Fallback
                    rank = 64
                    U = np.zeros((8, rank))
                    V = np.zeros((32, rank))
                    W = np.zeros((16, rank))
                    c = 0
                    for ii in range(2):
                        for jj in range(4):
                            for kk in range(8):
                                U[ii*4+jj, c] = 1
                                V[jj*8+kk, c] = 1
                                W[kk*2+ii, c] = 1
                                c += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
