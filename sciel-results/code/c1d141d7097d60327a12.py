import numpy as np
import sys

def construct():
    n, m, p = 4, 4, 7
    # Output W has shape (p*n, rank) with index W[k*n + i, r]
    # A is n x m (4x4), B is m x p (4x7), C is n x p (4x7)
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    terms = []  # list of (u_idx, v_idx, w_idx, coeff)
    
    def add_terms(u_idx, v_idx, w_idx, coeff):
        if abs(coeff) > 1e-12:
            terms.append((u_idx, v_idx, w_idx, coeff))
            
    # We split B into two parts:
    # Part 1: k in {0, 1, 2, 3} (first 4 columns of C). This is 4x4x4 multiplication.
    # Part 2: k in {4, 5, 6} (last 3 columns of C). This is 4x4x3 multiplication.
    
    # PART 1: 4x4x4 using Strassen (rank 49)
    # We can use Strassen decomposition for 4x4.
    # Strassen for 2x2 has rank 7.
    # For 4x4, we can use one level of Strassen where sub-blocks are 2x2.
    # The 7 products are 2x2 matrices.
    # Each 2x2 matrix product can be computed with rank 7 (Strassen) or rank 8 (schoolbook).
    # Using Strassen for sub-blocks: 7 * 7 = 49 rank.
    # Using schoolbook for sub-blocks: 7 * 8 = 56 rank.
    # Let's use the rank 49 method for better score.
    
    # Strassen 4x4 formulas:
    # A = [[A00, A01], [A10, A11]] where Ai are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bj are 2x2.
    
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
    
    # We need to expand these into scalar terms.
    # Each Mi is a product of two 2x2 matrices.
    # Let's write a helper to add the terms for a product (X + aY)(Z + bW) where X,Y,Z,W are 2x2 matrices (conceptually)
    # Actually, it's easier to just iterate.
    
    # Helper to add 2x2 matrix multiplication terms using Strassen (rank 7)
    # A2x2 is accessed via offsets in the global 4x4 A matrix.
    # B2x2 is accessed via offsets in the global 4x4 B matrix (for k=0..3).
    # C2x2 is accessed via offsets in the global 4x4 C matrix (for k=0..3).
    
    # However, the Strassen formulas above combine A and B blocks *before* multiplication.
    # e.g., M1 = (A00 + A11)(B00 + B11).
    # Let X = A00 + A11. This is a 2x2 matrix.
    # Let Y = B00 + B11. This is a 2x2 matrix.
    # We need to compute X * Y using Strassen (7 mults) or schoolbook (8 mults).
    
    # Let's use schoolbook for the inner 2x2 products first to ensure correctness, rank 56 for Part 1.
    # 56 + 48 (Part 2 schoolbook 4*4*3) = 104.
    # Can we do better?
    # If we use Strassen for inner 2x2 products: 7 * 7 = 49.
    # 49 + 48 = 97.
    
    # Let's implement the rank 49 solution.
    
    def add_2x2_product_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, sign_c, scale_c):
        # a_row_off, a_col_off: top-left (i,j) of the 2x2 block in A (4x4)
        # b_row_off, b_col_off: top-left (j,k) of the 2x2 block in B (4x4 for k<4)
        # c_row_off, c_col_off: top-left (i,k) of the 2x2 block in C (4x4 for k<4)
        # This function adds terms for the 2x2 matrix multiplication P[i,j] * Q[k,l] -> R[i,l]
        # P is formed from A at offset (a_row_off, a_col_off)
        # Q is formed from B at offset (b_row_off, b_col_off)
        # R is accumulated into C at offset (c_row_off, c_col_off) with sign_c and scale_c
        
        # The entries of P (2x2) correspond to global A indices:
        # P[0,0] = A[a_row_off, a_col_off]
        # P[0,1] = A[a_row_off, a_col_off+1]
        # P[1,0] = A[a_row_off+1, a_col_off]
        # P[1,1] = A[a_row_off+1, a_col_off+1]
        
        # Similarly for Q (2x2) from B:
        # Q[0,0] = B[b_row_off, b_col_off]
        # ...
        
        # And R (2x2) into C:
        # R[0,0] = C[c_row_off, c_col_off]
        # ...
        
        # Strassen 2x2:
        # M1 = (P00+P11)(Q00+Q11)
        # M2 = (P10+P11)Q00
        # M3 = P00(Q01-Q11)
        # M4 = P11(Q10-Q00)
        # M5 = (P00+P01)Q11
        # M6 = (P10-P00)(Q00+Q01)
        # M7 = (P01-P11)(Q10+Q11)
        
        # R00 = M1 + M4 - M5 + M7
        # R01 = M3 + M5
        # R10 = M2 + M4
        # R11 = M1 - M2 + M3 + M6
        
        # We need to expand each M into scalar products A[...] * B[...]
        
        # Helper to add a product of linear combinations
        # L(A_indices_with_coefs) * L(B_indices_with_coefs) -> C_target with coef
        
        # This is getting complex to code generically with offsets.
        # Let's just hardcode the 49 terms for the 4x4 case?
        # Or use a simpler rank 56 approach which is easier to verify.
        
        # Let's try rank 56 first (schoolbook inner).
        pass

    # Let's stick to the robust rank 104 approach (Schoolbook for inner 2x2) first to ensure 0 errors,
    # then optimize to rank 97 if time permits or if I can code the Strassen inner easily.
    # Actually, let's just do the full schoolbook for 4x4x4 (rank 64) and 4x4x3 (rank 48). Total 112.
    # Wait, the prompt says "Find a rank-minimal...". And "score is the rank".
    # The current program has score 0.0 because it returns empty matrices (rank 0? or invalid?).
    # The interface says score is the rank. Lower is better.
    # If I return 112, it's valid.
    # If I return 104, it's better.
    # If I return 97, even better.
    
    # Let's implement Rank 104 (Strassen outer, Schoolbook inner).
    
    # Helper: add schoolbook terms for 2x2 matrices at specific offsets
    def add_2x2_schoolbook(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k, coef):
        # A block at (a_off_i, a_off_j)
        # B block at (b_off_j, b_off_k)
        # C block at (c_off_i, c_off_k)
        # C[i,k] += coef * sum_j A[i,j]B[j,k]
        
        # Local indices 0..1
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    # A local (di, dj) -> global A (a_off_i+di, a_off_j+dj)
                    # B local (dj, dk) -> global B (b_off_j+dj, b_off_k+dk)
                    # C local (di, dk) -> global C (c_off_i+di, c_off_k+dk)
                    
                    # Wait, B is (m, p) = (4, 7).
                    # In the 4x4 part, k is 0..3.
                    # b_off_j is the row in B (0..3). b_off_k is the col in B (0..3).
                    # This is valid for the first 4 cols.
                    
                    u_idx = (a_off_i + di) * m + (a_off_j + dj)
                    v_idx = (b_off_j + dj) * p + (b_off_k + dk)
                    w_idx = (c_off_k + dk) * n + (c_off_i + di)  # W[k*n + i]
                    
                    add_terms(u_idx, v_idx, w_idx, coef)

    # Strassen 4x4 setup
    # A blocks:
    # A00: i 0-1, j 0-1 -> (0,0)
    # A01: i 0-1, j 2-3 -> (0,2)
    # A10: i 2-3, j 0-1 -> (2,0)
    # A11: i 2-3, j 2-3 -> (2,2)
    
    # B blocks (for k=0..3):
    # B00: j 0-1, k 0-1 -> (0,0)
    # B01: j 0-1, k 2-3 -> (0,2)
    # B10: j 2-3, k 0-1 -> (2,0)
    # B11: j 2-3, k 2-3 -> (2,2)
    
    # C blocks (for k=0..3):
    # C00: i 0-1, k 0-1 -> (0,0)
    # C01: i 0-1, k 2-3 -> (0,2)
    # C10: i 2-3, k 0-1 -> (2,0)
    # C11: i 2-3, k 2-3 -> (2,2)
    
    # M1 = (A00+A11)(B00+B11)
    # This expands to:
    # (A00+A11) B00 + (A00+A11) B11
    # = A00 B00 + A11 B00 + A00 B11 + A11 B11
    
    # Each term is a 2x2x2 multiplication.
    # We add 4 schoolbook products for M1.
    # M1 contributes to C00 and C11.
    # C00 += M1
    # C11 += M1 (in the formula C11 = M1 - M2 + M3 + M6)
    
    # So for each of the 4 sub-products in M1, we add terms to C00 (coef 1) and C11 (coef 1).
    
    # Let's define a helper to add a Strassen "product" which is a sum of A-blocks times a sum of B-blocks.
    def add_strassen_product(a_blocks, b_blocks, c_target_blocks, sign=1):
        # a_blocks: list of (off_i, off_j, coef)
        # b_blocks: list of (off_j, off_k, coef)
        # c_target_blocks: list of (off_i, off_k, coef) -- where the result is added
        # The product is (Sum ai * A_ai) * (Sum bi * B_bi)
        # = Sum ai bi A_ai B_bi
        # Each A_ai B_bi is added to the corresponding C block.
        
        for ai, bj, ca in a_blocks:
            for bk, ck, cb in b_blocks:
                total_coef = sign * ca * cb
                # A is at (ai, bj)? No, a_blocks are A offsets.
                # b_blocks are B offsets.
                # The A block is at (a_i, a_j). The B block is at (b_j, b_k).
                # The indices must match: a_j must equal b_j?
                # In Strassen, the blocks are aligned.
                # e.g. A00 (j=0..1) multiplies B00 (j=0..1). Yes.
                # A00 (j=0..1) multiplies B11 (j=2..3)? No!
                # Wait. Matrix multiplication C = A * B.
                # C_i,k = sum_j A_i,j B_j,k.
                # In block multiplication:
                # C00 = A00 B00 + A01 B10.
                # Strassen rewrites this.
                # M1 = (A00+A11)(B00+B11).
                # (A00+A11) is a 2x2 matrix.
                # (B00+B11) is a 2x2 matrix.
                # Their product is a 2x2 matrix.
                # But A00 is at (i 0-1, j 0-1).
                # B11 is at (j 2-3, k 2-3).
                # You can't just multiply A00 and B11 directly in the schoolbook sense because the 'j' indices don't overlap?
                # No, the "product" (A00+A11)(B00+B11) is computed as a 2x2 matrix multiplication.
                # Let X = A00+A11. X[0,0]=A00[0,0]+A11[0,0], etc.
                # Let Y = B00+B11. Y[0,0]=B00[0,0]+B11[0,0], etc.
                # M1 = X * Y.
                # M1[0,0] = X00 Y00 + X01 Y10.
                # X00 = A00[0,0] + A11[0,0]. (Indices in A: 0,0 and 2,2)
                # Y00 = B00[0,0] + B11[0,0]. (Indices in B: 0,0 and 2,2)
                # This is correct.
                # So the "schoolbook" expansion of M1 involves terms like (A00[0,0]+A11[0,0]) * (B00[0,0]+B11[0,0]).
                # This expands to 4 scalar products:
                # A00[0,0]B00[0,0] + A00[0,0]B11[0,0] + A11[0,0]B00[0,0] + A11[0,0]B11[0,0].
                
                # So, to add the terms for a Strassen product, we should:
                # 1. Identify the linear combination of A blocks.
                # 2. Identify the linear combination of B blocks.
                # 3. Expand all pairs.
                
                pass

    # This logic is too complicated to implement blindly without a clear data structure for the linear combos.
    
    # Let's go with the safest, most direct approach:
    # **Rank 112 Schoolbook** for the whole thing? No, 4x4x7 is 112.
    # But we can do better.
    
    # Let's use **Rank 97** (Strassen outer 4x4 + Schoolbook outer 4x3) IF I can implement Strassen outer easily.
    # Actually, the simplest improvement is **Rank 104** (Strassen outer 4x4 + Schoolbook inner 2x2).
    # Wait, if I use Strassen for the outer 4x4 (7 products of 2x2 matrices), and then use Schoolbook for each 2x2 matrix product (8 multiplications), the total rank is 7 * 8 = 56.
    # Plus the 4x4x3 part: 4 * 4 * 3 = 48.
    # Total 104.
    
    # Let's implement Rank 104.
    # I will explicitly generate the 56 terms for the 4x4 part.
    
    # Strassen 4x4 Linear Combinations:
    # M1: A: {00:1, 11:1}, B: {00:1, 11:1}
    # M2: A: {10:1, 11:1}, B: {00:1}
    # M3: A: {00:1}, B: {01:1, 11:-1}
    # M4: A: {11:1}, B: {10:1, 00:-1}
    # M5: A: {00:1, 01:1}, B: {11:1}
    # M6: A: {10:1, 00:-1}, B: {00:1, 01:1}
    # M7: A: {01:1, 11:-1}, B: {10:1, 11:1}
    
    # Offsets for blocks (2x2) in 4x4 grid:
    # 00: (0,0)
    # 01: (0,2)
    # 10: (2,0)
    # 11: (2,2)
    
    # C contributions (signs):
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    # For each M, we have linear combos of A and B.
    # We expand (Sum ca * A_ai) * (Sum cb * B_bi) into sum of scalar products.
    # Each scalar product A[ai_row, ai_col] * B[bi_row, bi_col] contributes to C.
    # Which C?
    # The result of M is a 2x2 matrix.
    # M1 is a 2x2 matrix. It adds to C00 (with sign 1) and C11 (with sign 1).
    # Wait, the formula C00 = M1 + ... means the 2x2 matrix M1 is added to the 2x2 block C00.
    # So M1[0,0] adds to C00[0,0]. M1[1,1] adds to C00[1,1].
    # AND M1[0,0] adds to C11[0,0]. M1[1,1] adds to C11[1,1].
    
    # So, for a given M, and a specific scalar product term A[i_a, j_a] * B[j_b, k_b] (where i_a,j_a are local in the A-block? No, they are local in the 2x2 structure).
    
    # Let's define the expansion more carefully.
    # For a Strassen term M, defined by A_combos and B_combos.
    # A_combos: list of (block_name, coef)
    # B_combos: list of (block_name, coef)
    
    # The 2x2 result matrix M has entries M[r, c] where r,c in 0..1.
    # M[r, c] = sum_{p in A_combos} sum_{q in B_combos} coef_p * coef_q * (A_block_p * B_block_q)[r, c]
    
    # (A_block_p * B_block_q)[r, c] is the (r,c) entry of the product of two 2x2 matrices.
    # But wait, A_block_p and B_block_q are 2x2 matrices themselves.
    # In the Strassen algorithm, we compute the product of the 2x2 matrices.
    # So if we use Schoolbook for the inner 2x2 product:
    # (A * B)[r, c] = sum_{k=0..1} A[r, k] * B[k, c].
    
    # So, M[r, c] = sum_p sum_q ca_p cb_q sum_{k=0..1} A_block_p[r, k] * B_block_q[k, c].
    
    # This means we have scalar terms of the form:
    # A_global[i_a, j_a] * B_global[j_b, k_b]
    # Where:
    # A_block_p is at offset (ip, jp).
    # A_block_p[r, k] corresponds to global A[i, j] = [ip + r, jp + k].
    # B_block_q is at offset (iq, jq).
    # B_block_q[k, c] corresponds to global B[i, k] = [iq + k, jq + c].
    
    # The resulting scalar product contributes to the 2x2 result M at position [r, c].
    # This M block is then added to the target C blocks.
    # If M is added to C00 (offset 0,0) with sign s, then:
    # Global C[i, k] += s * M[r, c] where i = 0 + r, k = 0 + c.
    # If M is added to C11 (offset 2,2) with sign s, then:
    # Global C[i, k] += s * M[r, c] where i = 2 + r, k = 2 + c.
    
    # Okay, this is implementable.
    
    # 1. Define blocks and offsets
    A_blocks = {
        '00': (0, 0),
        '01': (0, 2),
        '10': (2, 0),
        '11': (2, 2)
    }
    
    # Note: B blocks for the 4x4 part (k=0..3)
    B_blocks_4x4 = {
        '00': (0, 0), # j=0..1, k=0..1
        '01': (0, 2), # j=0..1, k=2..3
        '10': (2, 0), # j=2..3, k=0..1
        '11': (2, 2)  # j=2..3, k=2..3
    }
    
    # C blocks for the 4x4 part (k=0..3)
    C_blocks_4x4 = {
        '00': (0, 0),
        '01': (0, 2),
        '10': (2, 0),
        '11': (2, 2)
    }
    
    # Strassen definitions
    # M_name: (A_combos, B_combos, [ (C_block, sign), ... ])
    strassen_defs = [
        ('M1', [('00', 1), ('11', 1)], [('00', 1), ('11', 1)], [('00', 1), ('11', 1)]),
        ('M2', [('10', 1), ('11', 1)], [('00', 1)], [('10', 1), ('11', -1)]),
        ('M3', [('00', 1)], [('01', 1), ('11', -1)], [('01', 1), ('11', 1)]),
        ('M4', [('11', 1)], [('10', 1), ('00', -1)], [('00', 1), ('10', 1)]),
        ('M5', [('00', 1), ('01', 1)], [('11', 1)], [('00', -1), ('01', 1)]),
        ('M6', [('10', 1), ('00', -1)], [('00', 1), ('01', 1)], [('11', 1)]),
        ('M7', [('01', 1), ('11', -1)], [('10', 1), ('11', 1)], [('00', 1)])
    ]
    
    # Wait, let's double check the C signs.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: C00(+1), C11(+1) -> OK
    # M2: C10(+1), C11(-1) -> OK
    # M3: C01(+1), C11(+1) -> OK
    # M4: C00(+1), C10(+1) -> OK
    # M5: C00(-1), C01(+1) -> OK
    # M6: C11(+1) -> OK
    # M7: C00(+1) -> OK
    
    # Implementation of adding terms for one Strassen M
    for m_name, a_combs, b_combs, c_targets in strassen_defs:
        # Iterate over all pairs of A and B combos
        for (a_name, ca) in a_combs:
            a_off_i, a_off_j = A_blocks[a_name]
            for (b_name, cb) in b_combs:
                b_off_j, b_off_k = B_blocks_4x4[b_name]
                
                # The product is a 2x2 matrix.
                # We use schoolbook for the inner 2x2 product.
                # M[r, c] = sum_{k=0..1} A[r, k] * B[k, c]
                for r in range(2):
                    for c in range(2):
                        # The terms contributing to M[r,c] come from k=0 and k=1
                        for k_inner in range(2):
                            # A index:
                            # A_block is at (a_off_i, a_off_j).
                            # Local index [r, k_inner] -> Global A[i, j]
                            i_a = a_off_i + r
                            j_a = a_off_j + k_inner
                            u_idx = i_a * m + j_a
                            
                            # B index:
                            # B_block is at (b_off_j, b_off_k).
                            # Local index [k_inner, c] -> Global B[j, k]
                            # Wait, B_block row is j. In Strassen, the A block and B block must have matching "inner" dimensions.
                            # A is 2x2 (i,j). B is 2x2 (j,k).
                            # The local indices must be consistent.
                            # A[r, k_inner] and B[k_inner, c].
                            # Global A: i = a_off_i + r, j = a_off_j + k_inner.
                            # Global B: j = b_off_j + k_inner, k = b_off_k + c.
                            # Note: b_off_j should correspond to a_off_j + 2?
                            # No, the offsets are absolute.
                            # If A_name is '00' (j=0..1) and B_name is '11' (j=2..3), they don't multiply directly in standard matrix mult.
                            # BUT in Strassen, we multiply the *sums* of blocks.
                            # (A00+A11) * (B00+B11).
                            # This is not (A00*B00 + A00*B11 + ...) in the sense of disjoint indices.
                            # It is computing the matrix product of the 2x2 matrix (A00+A11) and the 2x2 matrix (B00+B11).
                            # So the "j" index in the multiplication is the local index k_inner (0 or 1).
                            # It does NOT refer to the global j index of the block.
                            # The global j index is determined by the block offset.
                            # However, for the multiplication to be valid, the "inner" dimension of A (which is j) and the "inner" dimension of B (which is j) must be aligned in the linear combination space.
                            # Strassen works because we are multiplying the *matrices* X and Y.
                            # X = A00 + A11. Y = B00 + B11.
                            # X[i, j] and Y[j, k] use local i,j,k (0..1).
                            # The global indices are just offsets.
                            # So, my loop above is correct:
                            # u_idx uses global A(i, j)
                            # v_idx uses global B(j, k)
                            
                            i_b = b_off_j + k_inner
                            k_b = b_off_k + c
                            v_idx = i_b * p + k_b
                            
                            # C targets
                            for c_name, c_sign in c_targets:
                                c_off_i, c_off_k = C_blocks_4x4[c_name]
                                # The result M[r, c] is placed at local [r, c] in the C block.
                                # Global C index:
                                i_c = c_off_i + r
                                k_c = c_off_k + c
                                w_idx = k_c * n + i_c
                                
                                # Coefficient
                                coeff = ca * cb * c_sign
                                add_terms(u_idx, v_idx, w_idx, coeff)
    
    # PART 2: 4x4x3 Schoolbook
    # k in {4, 5, 6}
    # C[i, k] = sum_j A[i, j] * B[j, k]
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_terms(u_idx, v_idx, w_idx, 1.0)
                
    # Construct matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx, r] = coeff
        V[v_idx, r] = coeff
        W[w_idx, r] = coeff
        
    return {"U": U, "V": V, "W": W}

# Quick self-test?
# No, just return.
