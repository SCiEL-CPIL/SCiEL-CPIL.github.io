"""Matrix multiplication 4x4 x 4x5 with rank-65 decomposition using Strassen."""
DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build U (16 x rank), V (20 x rank), W (20 x rank)
    # First 49 terms: Strassen for 4x4 x 4x4 (columns 0-3 of B)
    # Next 16 terms: scalar products for column 4 of B
    
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # A is n x m (4x4), B is m x p (4x5), C is n x p (4x5)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Strassen decomposition for 4x4 x 4x4 part
    # Partition A into 2x2 blocks of 2x2:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # Strassen 7 products for 2x2 blocks, each product is a 2x2x2 multiplication
    # which itself can be done with 7 rank-1 terms (or we can use schoolbook for 2x2x2 = 8 terms)
    # But to get 49 total, we use 7*7 = 49 by applying Strassen recursively
    
    # Let me define the 7 Strassen combinations for 2x2x2:
    # For multiplying 2x2 matrix P by 2x2 matrix Q to get 2x2 result R:
    # R[i,k] = sum_j P[i,j] * Q[j,k]
    
    # Strassen's 7 terms for 2x2x2:
    # M1 = (P[0,0]+P[1,1]) * (Q[0,0]+Q[1,1])
    # M2 = (P[1,0]+P[1,1]) * Q[0,0]
    # M3 = P[0,0] * (Q[0,1]-Q[1,1])
    # M4 = P[1,1] * (Q[1,0]-Q[0,0])
    # M5 = (P[0,0]+P[0,1]) * Q[1,1]
    # M6 = (P[1,0]-P[0,0]) * (Q[0,0]+Q[0,1])
    # M7 = (P[0,1]-P[1,1]) * (Q[1,0]+Q[1,1])
    
    # R[0,0] = M1 + M4 - M5 + M7
    # R[0,1] = M3 + M5
    # R[1,0] = M2 + M4
    # R[1,1] = M1 - M2 + M3 + M6
    
    # For each of the 7 Strassen products at the 4x4 level, we need to expand
    # each into 7 rank-1 terms for the 2x2x2 sub-problem.
    
    # Let me define the U, V, W factors for each of the 7 Strassen terms at 2x2x2 level.
    # For a 2x2x2 multiplication with indices (i,j,k) where i,k in {0,1}, j in {0,1}:
    # The 7 rank-1 terms have the following U (2x2), V (2x2), W (2x2) factors:
    
    strassen_2x2 = []
    # Each entry: (U_factor, V_factor, W_factor) where each is a dict mapping (idx) to coeff
    # U_factor: maps (i,j) -> coeff for the 2x2 A-block
    # V_factor: maps (j,k) -> coeff for the 2x2 B-block  
    # W_factor: maps (k,i) -> coeff for the 2x2 C-block
    
    # M1: (P00+P11)*(Q00+Q11), contributes to R00 (+1), R11 (+1)
    strassen_2x2.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2: (P10+P11)*Q00, contributes to R10 (+1), R11 (-1)
    strassen_2x2.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(0,1): 1, (1,1): -1}
    })
    # M3: P00*(Q01-Q11), contributes to R01 (+1), R11 (+1)
    strassen_2x2.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(1,0): 1, (1,1): 1}
    })
    # M4: P11*(Q10-Q00), contributes to R00 (+1), R10 (+1)
    strassen_2x2.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (0,1): 1}
    })
    # M5: (P00+P01)*Q11, contributes to R00 (-1), R01 (+1)
    strassen_2x2.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (1,0): 1}
    })
    # M6: (P10-P00)*(Q00+Q01), contributes to R11 (+1)
    strassen_2x2.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7: (P01-P11)*(Q10+Q11), contributes to R00 (+1)
    strassen_2x2.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    # Now for 4x4 x 4x4, we have 7 Strassen combinations at the top level.
    # Each top-level combination involves linear combinations of the 2x2 blocks.
    # Let me define the 7 top-level Strassen terms for 4x4:
    
    # Top-level Strassen for 4x4x4:
    # M1_top = (A11+A22)(B11+B22)
    # M2_top = (A21+A22)B11
    # M3_top = A11(B12-B22)
    # M4_top = A22(B21-B11)
    # M5_top = (A11+A12)B22
    # M6_top = (A21-A11)(B11+B12)
    # M7_top = (A12-A22)(B21+B22)
    
    # Each of these is a 2x2 x 2x2 multiplication (of the block matrices),
    # which we decompose using the 7 Strassen terms above.
    
    # Block index mapping:
    # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1) where block (bi,bj) covers rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
    # Similarly for B and C.
    
    # For each top-level Strassen term, I need to specify:
    # - Which A blocks are combined and with what coefficients
    # - Which B blocks are combined and with what coefficients
    # - How the resulting C blocks are combined
    
    # Let me define the 7 top-level terms more carefully:
    
    top_strassen = []
    # Each entry: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # A_block_coeffs: dict from (bi,bj) -> coeff
    # B_block_coeffs: dict from (bj,bk) -> coeff  
    # C_block_coeffs: dict from (bk,bi) -> coeff
    
    # M1_top = (A11+A22)(B11+B22)
    # This is a product of two 2x2 block matrices. We decompose it with 7 Strassen sub-terms.
    # For the decomposition, we treat the "P" matrix as having blocks with coefficients from A11+A22
    # and "Q" matrix as having blocks from B11+B22.
    
    # Actually, let me think differently. The top-level Strassen gives 7 scalar products
    # where each product is between a linear combination of A elements and a linear combination of B elements.
    # But since we're working with 2x2 blocks, each "scalar product" is actually a 2x2x2 tensor.
    
    # Let me just directly construct the 49 rank-1 terms.
    
    # For top-level term t (0..6), and sub-level term s (0..6):
    # The U factor combines A elements, V factor combines B elements, W factor distributes to C elements.
    
    # Top-level Strassen definitions (which A blocks and B blocks are multiplied):
    # t=0: P = A11+A22, Q = B11+B22
    # t=1: P = A21+A22, Q = B11
    # t=2: P = A11, Q = B12-B22
    # t=3: P = A22, Q = B21-B11
    # t=4: P = A11+A12, Q = B22
    # t=5: P = A21-A11, Q = B11+B12
    # t=6: P = A12-A22, Q = B21+B22
    
    # For each top-level term, P is a 2x2 block matrix where each block is a linear combo of A blocks.
    # Q is a 2x2 block matrix where each block is a linear combo of B blocks.
    
    # Let me define P_blocks[t] = dict from (bi,bj) -> coeff, representing the 2x2 block matrix P
    # and Q_blocks[t] = dict from (bj,bk) -> coeff, representing the 2x2 block matrix Q
    
    P_blocks = [
        {(0,0): 1, (1,1): 1},  # t=0: A11+A22
        {(1,0): 1, (1,1): 1},  # t=1: A21+A22
        {(0,0): 1},             # t=2: A11
        {(1,1): 1},             # t=3: A22
        {(0,0): 1, (0,1): 1},  # t=4: A11+A12
        {(1,0): 1, (0,0): -1}, # t=5: A21-A11
        {(0,1): 1, (1,1): -1}, # t=6: A12-A22
    ]
    
    Q_blocks = [
        {(0,0): 1, (1,1): 1},  # t=0: B11+B22
        {(0,0): 1},             # t=1: B11
        {(0,1): 1, (1,1): -1}, # t=2: B12-B22
        {(1,0): 1, (0,0): -1}, # t=3: B21-B11
        {(1,1): 1},             # t=4: B22
        {(0,0): 1, (0,1): 1},  # t=5: B11+B12
        {(1,0): 1, (1,1): 1},  # t=6: B21+B22
    ]
    
    # Now for each top-level term t and sub-level term s, we get one rank-1 term.
    # The sub-level Strassen term s has:
    #   U_sub[s]: dict from (i,j) -> coeff for the 2x2 P matrix
    #   V_sub[s]: dict from (j,k) -> coeff for the 2x2 Q matrix
    #   W_sub[s]: dict from (k,i) -> coeff for the 2x2 R matrix
    
    # The full U factor for term (t,s):
    # For each (i,j) in U_sub[s] with coeff cu, and for each A block (bi,bj) in P_blocks[t] with coeff cp:
    #   The A element at global index (2*bi+i, 2*bj+j) gets coefficient cu * cp
    
    # Similarly for V and W.
    
    # Let me build this:
    
    term_col = 0
    for t in range(7):
        for s in range(7):
            col = term_col
            term_col += 1
            
            # Build U factor
            for (si, sj), cu in strassen_2x2[s]['U'].items():
                for (bi, bj), cp in P_blocks[t].items():
                    global_i = 2 * bi + si
                    global_j = 2 * bj + sj
                    U[global_i * m + global_j][col] += cu * cp
            
            # Build V factor
            for (sj, sk), cv in strassen_2x2[s]['V'].items():
                for (bj, bk), cq in Q_blocks[t].items():
                    global_j = 2 * bj + sj
                    global_k = 2 * bk + sk
                    # Only for k in 0..3 (first 4 columns of B)
                    if global_k < 4:
                        V[global_j * p + global_k][col] += cv * cq
            
            # Build W factor
            for (sk, si), cw in strassen_2x2[s]['W'].items():
                # C block (bk, bi) where bk = sk//2? No...
                # The sub-level W maps (k,i) within the 2x2 block to the result.
                # The top-level result is distributed to C blocks.
                # Actually, the W factor at sub-level gives coefficients for the 2x2 result block.
                # The top-level doesn't add more W coefficients; the W is just the sub-level W.
                # Wait, I need to think about this more carefully.
                
                # The sub-level term computes a 2x2 result R_sub where:
                # R_sub[si,sk] gets coefficient cw from this term.
                # But R_sub corresponds to a specific C block.
                
                # Actually, I think I'm overcomplicating this. Let me reconsider.
                
                # The top-level Strassen computes C blocks as linear combinations of the 7 top-level products.
                # Each top-level product is itself a 2x2x2 tensor (the product of two 2x2 block matrices).
                # So the W factor should incorporate both the top-level distribution and the sub-level distribution.
                
                # Let me define the top-level W distribution:
                # C11 = M1 + M4 - M5 + M7  -> C block (0,0)
                # C12 = M3 + M5            -> C block (0,1)
                # C21 = M2 + M4            -> C block (1,0)
                # C22 = M1 - M2 + M3 + M6  -> C block (1,1)
                
                # So for top-level term t, it contributes to certain C blocks with certain coefficients.
                # Let me define W_top[t] = dict from (bk,bi) -> coeff
                
                pass  # Will fix below
    
    # Let me redo this more carefully with the top-level W distribution.
    
    # Reset
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Top-level W distribution: which C blocks each top-level term contributes to
    W_top = [
        {(0,0): 1, (1,1): 1},       # t=0 (M1): C11 += 1, C22 += 1
        {(1,0): 1, (1,1): -1},      # t=1 (M2): C21 += 1, C22 -= 1
        {(0,1): 1, (1,1): 1},       # t=2 (M3): C12 += 1, C22 += 1
        {(0,0): 1, (1,0): 1},       # t=3 (M4): C11 += 1, C21 += 1
        {(0,0): -1, (0,1): 1},      # t=4 (M5): C11 -= 1, C12 += 1
        {(1,1): 1},                 # t=5 (M6): C22 += 1
        {(0,0): 1},                 # t=6 (M7): C11 += 1
    ]
    
    term_col = 0
    for t in range(7):
        for s in range(7):
            col = term_col
            term_col += 1
            
            # Build U factor: combine sub-level U with top-level P blocks
            for (si, sj), cu in strassen_2x2[s]['U'].items():
                for (bi, bj), cp in P_blocks[t].items():
                    global_i = 2 * bi + si
                    global_j = 2 * bj + sj
                    U[global_i * m + global_j][col] += cu * cp
            
            # Build V factor: combine sub-level V with top-level Q blocks
            for (sj, sk), cv in strassen_2x2[s]['V'].items():
                for (bj, bk), cq in Q_blocks[t].items():
                    global_j = 2 * bj + sj
                    global_k = 2 * bk + sk
                    if global_k < 4:  # Only first 4 columns of B
                        V[global_j * p + global_k][col] += cv * cq
            
            # Build W factor: combine sub-level W with top-level W distribution
            for (sk, si), cw in strassen_2x2[s]['W'].items():
                for (bk, bi), ct in W_top[t].items():
                    global_k = 2 * bk + sk
                    global_i = 2 * bi + si
                    W[global_k * n + global_i][col] += cw * ct
    
    # Now add 16 scalar terms for the 5th column (k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # For each (i,j) pair, we have a rank-1 term:
    # U[i*m+j] = 1, V[j*p+4] = 1, W[4*n+i] = 1
    
    for i in range(n):
        for j in range(m):
            col = 49 + i * m + j
            U[i * m + j][col] = 1.0
            V[j * p + 4][col] = 1.0
            W[4 * n + i][col] = 1.0
    
    # Convert to lists of floats (already are)
    return {"U": U, "V": V, "W": W}

