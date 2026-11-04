"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Construct a rank-88 exact decomposition for 3x4 by 4x8 multiplication.
    # Strategy: split the 4x4 middle dimension into two 2x2 blocks.
    # Each 3x2x8 sub-problem is solved via splitting n=3 into 2+1,
    # p=8 into 4+4, and using Strassen's algorithm (rank 7) for the 2x2x4 part.
    # 3x2x8 rank: 2*(Strassen 2x2x4 rank) + 1*(schoolbook 2x4 rank)
    #           = 2*7 + 1*8 = 22? No wait.
    # Let's re-verify the 3x2x8 decomposition:
    # A is 3x2, B is 2x8. C = A*B is 3x8.
    # Split rows of A: A_top (2x2), A_bot (1x2).
    # Split cols of B: B_left (2x4), B_right (2x4).
    # C_top_left (2x4) = A_top * B_left. Rank 7 (Strassen).
    # C_top_right (2x4) = A_top * B_right. Rank 7 (Strassen).
    # C_bot_left (1x4) = A_bot * B_left. Rank 4 (schoolbook, 1x2*2x4 -> 4 mults? No, 1x4 result needs 4 mults? A_bot is 1x2, B_left is 2x4. Product is 1x4. Each of 4 entries is sum of 2 products. So 4*2=8 mults? No, 1x4 matrix multiplication by 1x2 and 2x4. The tensor rank is 1*2*4 = 8? No, rank is max dimension? No.
    # Actually, for 1x2 * 2x4, the product C_k = A_0 B_0k + A_1 B_1k. This is a sum of 2 rank-1 tensors (outer products of A_0 with row 0 of B_left, etc). So rank is 2? No, the tensors are in the space of A (1x2) and B (2x4) and C (1x4).
    # The multiplication tensor T_{i,j,k} = delta_{i,j} * delta_{j,k}? No.
    # Let's just use schoolbook for the 1x2x4 part: rank = 1*2*4 = 8?
    # Wait, the "rank" of 1x2x4 matrix multiplication:
    # C_k = A_0 B_0k + A_1 B_1k.
    # This requires 2 multiplications? No, A_0*B_0k is one mult for each k. There are 4 k's. So 4 mults for A_0 part, 4 mults for A_1 part?
    # No, A_0 is a scalar. B_0k is a scalar. A_0 * B_0k is one scalar mult. We do this for k=0..3. So 4 mults.
    # Similarly A_1 * B_1k for k=0..3. 4 mults.
    # Total 8 mults.
    # So 3x2x8 rank = 7 + 7 + 8 = 22.
    # Two such blocks: 22 + 22 = 44?
    # But we have two 3x2x8 blocks: one for j=0,1 and one for j=2,3.
    # Total rank 44.
    # This contradicts my previous estimate of 88. Why?
    # Ah, I see. The previous estimate was 44+44=88. But 44 is for the WHOLE 3x4x8?
    # Let's re-read: "3x2x8 rank is 22+22=44. Total for 3x4x8: 44+44=88."
    # Wait. 3x2x8 is ONE half of the 3x4x8 problem (splitting m=4 into 2+2).
    # So if 3x2x8 rank is 22, then 3x4x8 rank is 22+22=44?
    # Let's check the logic.
    # 3x4x8: A is 3x4, B is 4x8.
    # Split A into A1 (3x2) and A2 (3x2).
    # Split B into B1 (2x8) and B2 (2x8).
    # C = A1*B1 + A2*B2.
    # Rank(3x4x8) <= Rank(3x2x8) + Rank(3x2x8).
    # If Rank(3x2x8) = 22, then Rank(3x4x8) <= 44.
    # Is Rank(3x2x8) really 22?
    # 3x2x8: A is 3x2, B is 2x8.
    # Split A into A_top (2x2) and A_bot (1x2).
    # Split B into B_left (2x4) and B_right (2x4).
    # C = [A_top; A_bot] * [B_left B_right] = [[A_top*B_left, A_top*B_right], [A_bot*B_left, A_bot*B_right]].
    # Top-left (2x4): Rank 7 (Strassen).
    # Top-right (2x4): Rank 7 (Strassen).
    # Bottom-left (1x4): Rank 8 (Schoolbook 1x2*2x4).
    # Bottom-right (1x4): Rank 8 (Schoolbook 1x2*2x4).
    # Total rank = 7 + 7 + 8 + 8 = 30.
    # So 3x2x8 rank <= 30.
    # Then 3x4x8 rank <= 30 + 30 = 60.
    #
    # Can we do better?
    # Strassen for 2x2x4 has rank 7.
    # Schoolbook for 1x2x4 has rank 8.
    #
    # What if we split p=8 into 2+2+2+2?
    # 2x2x2 rank is 7 (Strassen).
    # 1x2x2 rank is 4 (Schoolbook).
    # 3x2x8:
    # Top part (2x2 A, 2x8 B): Split B into four 2x2 blocks.
    # Each 2x2x2 is rank 7. Four blocks -> 28.
    # Bottom part (1x2 A, 2x8 B): Split B into four 2x2 blocks.
    # Each 1x2x2 is rank 4. Four blocks -> 16.
    # Total 3x2x8 rank = 28 + 16 = 44.
    # This is worse than 30.
    #
    # So 3x2x8 rank is likely 30.
    # And 3x4x8 rank is likely 60.
    #
    # Let's try to find a better decomposition for 3x2x8.
    # Maybe combine the bottom parts?
    # A_bot is 1x2. B_left and B_right are 2x4.
    # C_bot = A_bot * [B_left B_right] = A_bot * B_full (where B_full is 2x8).
    # Rank of 1x2x8 is 1*2*8 = 16?
    # C_k = A_0 B_0k + A_1 B_1k.
    # This is 16 scalar multiplications?
    # A_0 is one scalar. B_0k is 8 scalars. A_0*B_0k is 8 mults.
    # A_1 is one scalar. B_1k is 8 scalars. A_1*B_1k is 8 mults.
    # Total 16 mults.
    # So Rank(1x2x8) = 16.
    #
    # So 3x2x8 rank <= Rank(2x2x8) + Rank(1x2x8).
    # Rank(2x2x8): Split B into 2x4 + 2x4.
    # Rank(2x2x4) = 7 (Strassen).
    # So Rank(2x2x8) <= 7 + 7 = 14.
    # Rank(3x2x8) <= 14 + 16 = 30.
    #
    # So 3x4x8 rank <= 30 + 30 = 60.
    #
    # Let's implement the rank-60 decomposition.
    
    n, m, p = DIMENSIONS
    
    # We will construct U, V, W matrices.
    # U shape: (n*m, rank) = (12, 60)
    # V shape: (m*p, rank) = (32, 60)
    # W shape: (p*n, rank) = (24, 60)
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1
    
    # Helper to add Strassen 2x2x4 block
    # A is 2x2, B is 2x4. C is 2x4.
    # We need to map local indices to global indices.
    # Let i0, j0, k0 be the starting indices for A rows, B cols, C cols.
    # A rows: i0, i0+1.
    # B cols: k0..k0+3.
    # B rows: j0, j0+1 (these are the middle dimension indices).
    # C cols: k0..k0+3.
    # C rows: i0, i0+1.
    
    def add_strassen_2x2x4(i0, j0, k0):
        nonlocal col
        # Strassen 2x2x2 has rank 7.
        # 2x2x4 is two 2x2x2 blocks?
        # B is 2x4. Split B into B1 (2x2) and B2 (2x2).
        # C1 = A * B1. C2 = A * B2.
        # Each is rank 7. Total 14.
        # Let's implement 2x2x2 Strassen.
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # But we are doing 2x2x4.
        # Let's just do two 2x2x2 Strassen calls.
        
        # Block 1: k0, k0+1
        # A rows i0, i0+1. B rows j0, j0+1. B cols k0, k0+1.
        # C rows i0, i0+1. C cols k0, k0+1.
        
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # This is not a single multiplication in the tensor decomposition sense.
        # The tensor decomposition uses bilinear forms.
        # The standard Strassen decomposition uses 7 multiplications of linear combinations.
        # We need to express the 7 products as rank-1 tensors.
        
        # Let's find the rank-1 tensors for Strassen 2x2x2.
        # The 7 multiplications are:
        # 1. (a00+a11)(b00+b11) -> U: a00, a11. V: b00, b11. W: c00, c11 (with signs)
        # This is not a single rank-1 term. It's a product of sums.
        # To decompose (x+y)(z+w) = xz + xw + yz + yw into rank-1 tensors, it takes 4 terms.
        # So Strassen's "7 multiplications" are not 7 rank-1 tensors.
        # They are 7 scalar multiplications of precomputed linear combinations.
        # The tensor rank is the number of scalar multiplications needed.
        # So yes, Strassen 2x2x2 has tensor rank 7.
        
        # How to represent the 7 rank-1 terms?
        # Term 1: U = [1, 0, 0, 1] (for a00, a10, a11, a11? No, A is 2x2 flat).
        # Let A = [[a00, a01], [a10, a11]]. Flat: a00, a01, a10, a11.
        # Let B = [[b00, b01], [b10, b11]]. Flat: b00, b01, b10, b11.
        # Let C = [[c00, c01], [c10, c11]]. Flat: c00, c01, c10, c11.
        
        # T1 = (a00 + a11)(b00 + b11)
        # This contributes to c00 and c11.
        # c00 += T1, c11 -= T1? No.
        # c00 = T1 + T4 - T5 + T7
        # c11 = T1 - T2 + T3 + T6
        
        # The rank-1 term for T1 is:
        # U1: [1, 0, 0, 1] (coeffs for a00, a01, a10, a11) -> a00 + a11
        # V1: [1, 0, 0, 1] (coeffs for b00, b01, b10, b11) -> b00 + b11
        # W1: [1, 0, 0, -1] (coeffs for c00, c01, c10, c11) -> c00 - c11
        # Wait, c00 gets +T1, c11 gets +T1.
        # So W1 should be [1, 0, 0, 1].
        
        # Let's list the 7 terms for Strassen 2x2x2.
        # 1. (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        # 2. (a10+a11)b00 -> c10 += 1, c11 -= 1
        # 3. a00(b01-b11) -> c01 += 1, c11 += 1
        # 4. a11(b10-b00) -> c00 += 1, c10 += 1
        # 5. (a00+a01)b11 -> c00 -= 1, c01 += 1
        # 6. (a10-a00)(b00+b01) -> c11 += 1, c01 += 1
        # 7. (a01-a11)(b10+b11) -> c00 += 1, c10 -= 1
        
        # Let's verify:
        # c00 = T1 + T4 - T5 + T7
        # c01 = T3 + T5 + T6
        # c10 = T2 + T4 - T7
        # c11 = T1 - T2 + T3 + T6
        
        # This looks correct.
        
        # Now, for 2x2x4:
        # We have B as 2x4.
        # We can view it as two 2x2 blocks.
        # B1 = cols 0,1. B2 = cols 2,3.
        # C1 = A*B1. C2 = A*B2.
        # Each uses 7 multiplications.
        # Total 14.
        
        # Let's add the 14 terms.
        
        # Block 1 (cols k0, k0+1)
        # U vectors (for A part, 2x2 -> 4 entries):
        # V vectors (for B part, 2x4 -> 8 entries, but we only use 2 cols)
        # W vectors (for C part, 2x4 -> 8 entries, but we only use 2 cols)
        
        # This is getting complicated with global indices.
        # Let's just use the schoolbook decomposition for simplicity first to ensure correctness, then optimize.
        # Schoolbook rank for 3x4x8 is 96.
        # We want to improve on 96.
        #
        # Let's try a simpler improvement: Strassen for 2x2x2 blocks.
        # Split 3x4x8 into:
        # Rows 0,1 of A (2x4), Row 2 of A (1x4).
        # Cols 0,1,2,3 of B (4x8).
        # 
        # C_top (2x8) = A_top (2x4) * B (4x8).
        # C_bot (1x8) = A_bot (1x4) * B (4x8).
        
        # C_bot: Rank = 1*4*8 = 32 (Schoolbook).
        
        # C_top: A_top is 2x4. B is 4x8.
        # Split B into four 4x2 blocks? Or B into 4 rows, each 1x8.
        # Let's split the middle dimension 4 into 2+2.
        # A_top = [A1 (2x2) A2 (2x2)]
        # B = [B1 (2x8); B2 (2x8)]
        # C_top = A1*B1 + A2*B2.
        # Rank(A1*B1) = Rank(2x2x8).
        # Rank(2x2x8) = 2 * Rank(2x2x4) = 2 * 7 = 14.
        # So Rank(C_top) = 14 + 14 = 28.
        
        # Total rank = 32 + 28 = 60.
        
        # This matches my previous calculation.
        
        # Now, let's implement the 2x2x8 decomposition with rank 14.
        # 2x2x8: A is 2x2, B is 2x8.
        # Split B into B1 (2x4) and B2 (2x4).
        # C1 = A*B1 (2x4). C2 = A*B2 (2x4).
        # Rank(2x2x4) = 7 (Strassen).
        
        # So we need a function to add a Strassen 2x2x4 decomposition.
        
        def add_strassen_2x2x4(i_start, j_start, k_start):
            nonlocal col
            # A rows: i_start, i_start+1
            # B rows: j_start, j_start+1
            # B cols: k_start .. k_start+3
            # C rows: i_start, i_start+1
            # C cols: k_start .. k_start+3
            
            # We do two 2x2x2 Strassen blocks.
            
            # Block 1: k_start, k_start+1
            # Block 2: k_start+2, k_start+3
            
            for block in range(2):
                kb = k_start + 2 * block
                
                # U vectors (2x2 A -> 4 entries)
                # V vectors (2x4 B -> 8 entries, but we only touch 2 cols in this block)
                # W vectors (2x4 C -> 8 entries, but we only touch 2 cols in this block)
                
                # The 7 Strassen terms for 2x2x2:
                # U (A part), V (B part), W (C part)
                # Indices: A: [0,1,2,3] for [a00, a01, a10, a11]
                #          B: [0,1,2,3] for [b00, b01, b10, b11]
                #          C: [0,1,2,3] for [c00, c01, c10, c11]
                
                # We map these to global indices.
                
                # A indices:
                # a00 -> i_start, j_start
                # a01 -> i_start, j_start+1
                # a10 -> i_start+1, j_start
                # a11 -> i_start+1, j_start+1
                
                # B indices:
                # b00 -> j_start, kb
                # b01 -> j_start, kb+1
                # b10 -> j_start+1, kb
                # b11 -> j_start+1, kb+1
                
                # C indices:
                # c00 -> i_start, kb
                # c01 -> i_start, kb+1
                # c10 -> i_start+1, kb
                # c11 -> i_start+1, kb+1
                
                # We define the 7 terms.
                # Each term is (U_vec, V_vec, W_vec) where vecs are sparse.
                # We will add them one by one.
                
                # Term 1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
                add_term(i_start * m + j_start, j_start * p + kb, kb * n + i_start, 1)
                add_term((i_start + 1) * m + (j_start + 1), (j_start + 1) * p + (kb + 1), (kb + 1) * n + (i_start + 1), 1)
                # Wait, this is not how it works.
                # The rank-1 term is U * V * W.
                # U is a vector in A-space, V in B-space, W in C-space.
                # The term contributes U[a] * V[b] * W[c] to T[a,b,c].
                # For Strassen term 1:
                # U = [1, 0, 0, 1] (a00 + a11)
                # V = [1, 0, 0, 1] (b00 + b11)
                # W = [1, 0, 0, 1] (c00 + c11)
                # This is NOT a rank-1 tensor in the original space.
                # It is a rank-1 tensor in the space of linear combinations.
                # To express it as rank-1 tensors in the original basis, we need to expand.
                # (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
                # This is 4 rank-1 terms.
                # So Strassen's "7 multiplications" are not 7 rank-1 terms.
                # The tensor rank is the minimum number of rank-1 terms.
                # Strassen's algorithm shows that 7 multiplications suffice, but these are multiplications of linear combinations.
                # Each such multiplication corresponds to a rank-1 term in the space of the linear combinations.
                # But in the original basis, the rank is higher?
                # No, the tensor rank is invariant.
                # The 7 multiplications are 7 bilinear forms.
                # Each bilinear form is a rank-1 tensor in the space of the input/output linear combinations.
                # But we can choose the basis of the input/output space arbitrarily.
                # So the tensor rank is indeed 7.
                
                # How to write the 7 rank-1 terms in the original basis?
                # We don't need to. We just need to provide U, V, W matrices.
                # The columns of U, V, W are the vectors for each rank-1 term.
                # So for Strassen term 1, the U vector is [1, 0, 0, 1] (in the local A basis).
                # The V vector is [1, 0, 0, 1] (in the local B basis).
                # The W vector is [1, 0, 0, 1] (in the local C basis).
                
                # So we just add the vectors.
                
                # Let's define the local basis vectors.
                # A local: 0:a00, 1:a01, 2:a10, 3:a11
                # B local: 0:b00, 1:b01, 2:b10, 3:b11
                # C local: 0:c00, 1:c01, 2:c10, 3:c11
                
                # Term 1
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                U_vec[i_start * m + j_start] = 1
                U_vec[(i_start + 1) * m + (j_start + 1)] = 1
                V_vec[j_start * p + kb] = 1
                V_vec[(j_start + 1) * p + (kb + 1)] = 1
                W_vec[kb * n + i_start] = 1
                W_vec[(kb + 1) * n + (i_start + 1)] = 1
                
                U[col] = U_vec
                V[col] = V_vec
                W[col] = W_vec
                col += 1
                
                # Term 2: (a10+a11)b00 -> c10 += 1, c11 -= 1
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                U_vec[(i_start + 1) * m + j_start] = 1
                U_vec[(i_start + 1) * m + (j_start + 1)] = 1
                V_vec[j_start * p + kb] = 1
                W_vec[kb * n + (i_start + 1)] = 1
                W_vec[(kb + 1) * n + (i_start + 1)] = -1
                
                U[col] = U_vec
                V[col] = V_vec
                W[col] = W_vec
                col += 1
                
                # Term 3: a00(b01-b11) -> c01 += 1, c11 += 1
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                U_vec[i_start * m + j_start] = 1
                V_vec[j_start * p + (kb + 1)] = 1
                V_vec[(j_start + 1) * p + (kb + 1)] = -1
                W_vec[(kb + 1) * n + i_start] = 1
                W_vec[(kb + 1) * n + (i_start + 1)] = 1
                
                U[col] = U_vec
                V[col] = V_vec
                W[col] = W_vec
                col += 1
                
                # Term 4: a11(b10-b00) -> c00 += 1, c10 += 1
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                U_vec[(i_start + 1) * m + (j_start + 1)] = 1
                V_vec[(j_start + 1) * p + kb] = 1
                V_vec[j_start * p + kb] = -1
                W_vec[kb * n + i_start] = 1
                W_vec[kb * n + (i_start + 1)] = 1
                
                U[col] = U_vec
                V[col] = V_vec
                W[col] = W_vec
                col += 1
                
                # Term 5: (a00+a01)b11 -> c00 -= 1, c01 += 1
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                U_vec[i_start * m + j_start] = 1
                U_vec[i_start * m + (j_start + 1)] = 1
                V_vec[(j_start + 1) * p + (kb + 1)] = 1
                W_vec[kb * n + i_start] = -1
                W_vec[(kb + 1) * n + i_start] = 1
                
                U[col] = U_vec
                V[col] = V_vec
                W[col] = W_vec
                col += 1
                
                # Term 6: (a10-a00)(b00+b01) -> c11 += 1, c01 += 1
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                U_vec[(i_start + 1) * m + j_start] = 1
                U_vec[i_start * m + j_start] = -1
                V_vec[j_start * p + kb] = 1
                V_vec[j_start * p + (kb + 1)] = 1
                W_vec[(kb + 1) * n + (i_start + 1)] = 1
                W_vec[(kb + 1) * n + i_start] = 1
                
                U[col] = U_vec
                V[col] = V_vec
                W[col] = W_vec
                col += 1
                
                # Term 7: (a01-a11)(b10+b11) -> c00 += 1, c10 -= 1
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                U_vec[i_start * m + (j_start + 1)] = 1
                U_vec[(i_start + 1) * m + (j_start + 1)] = -1
                V_vec[(j_start + 1) * p + kb] = 1
                V_vec[(j_start + 1) * p + (kb + 1)] = 1
                W_vec[kb * n + i_start] = 1
                W_vec[kb * n + (i_start + 1)] = -1
                
                U[col] = U_vec
                V[col] = V_vec
                W[col] = W_vec
                col += 1
        
        # Now, let's assemble the 3x4x8 decomposition.
        
        # C_bot: A_bot (1x4) * B (4x8). Rank 32.
        # A_bot rows: i=2.
        # B rows: j=0,1,2,3.
        # B cols: k=0..7.
        # C_bot rows: i=2.
        # C_bot cols: k=0..7.
        
        for j in range(4):
            for k in range(8):
                add_term(2 * m + j, j * p + k, k * n + 2, 1)
        
        # C_top: A_top (2x4) * B (4x8). Rank 28.
        # Split B into B1 (rows 0,1) and B2 (rows 2,3).
        # C_top = A_top * B1 + A_top * B2.
        
        # A_top * B1: A_top (2x4), B1 (2x8).
        # Split B1 into B1a (2x4, cols 0-3) and B1b (2x4, cols 4-7).
        # A_top * B1a: Rank 7 (Strassen 2x2x4).
        # A_top * B1b: Rank 7 (Strassen 2x2x4).
        
        add_strassen_2x2x4(0, 0, 0) # A rows 0,1. B rows 0,1. B cols 0,1.
        # Wait, add_strassen_2x2x4 handles 2x2 A and 2x4 B.
        # Here A is 2x4.
        # So we need to split A into A1 (2x2) and A2 (2x2).
        
        # A1 (rows 0,1, cols 0,1) * B1a (rows 0,1, cols 0-3)
        # This is 2x2 A and 2x4 B.
        # But B1a has 2 rows (j=0,1) and 4 cols (k=0..3).
        # So add_strassen_2x2x4(0, 0, 0) is correct for A cols 0,1 and B rows 0,1 and B cols 0,1.
        # But we need B cols 0..3.
        # So we call add_strassen_2x2x4(0, 0, 0) for B cols 0,1.
        # And add_strassen_2x2x4(0, 0, 2) for B cols 2,3.
        
        add_strassen_2x2x4(0, 0, 0) # A cols 0,1. B rows 0,1. B cols 0,1.
        add_strassen_2x2x4(0, 0, 2) # A cols 0,1. B rows 0,1. B cols 2,3.
        
        # A2 (rows 0,1, cols 2,3) * B1a (rows 0,1, cols 0-3)
        add_strassen_2x2x4(0, 2, 0) # A cols 2,3. B rows 0,1. B cols 0,1.
        add_strassen_2x2x4(0, 2, 2) # A cols 2,3. B rows 0,1. B cols 2,3.
        
        # A_top * B2: A_top (2x4), B2 (2x8).
        # B2 rows 2,3.
        add_strassen_2x2x4(0, 0, 4) # A cols 0,1. B rows 2,3. B cols 4,5.
        add_strassen_2x2x4(0, 0, 6) # A cols 0,1. B rows 2,3. B cols 6,7.
        add_strassen_2x2x4(0, 2, 4) # A cols 2,3. B rows 2,3. B cols 4,5.
        add_strassen_2x2x4(0, 2, 6) # A cols 2,3. B rows 2,3. B cols 6,7.
        
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
