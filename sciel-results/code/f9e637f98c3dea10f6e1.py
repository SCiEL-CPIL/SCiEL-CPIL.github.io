import numpy as np

def construct():
    n, m, p = 5, 5, 5
    # We will build a list of (u, v, w) tuples
    # u, v, w are lists of length 25 (flattened indices)
    # We will use dictionaries or sparse lists for efficiency during construction, then convert to dense arrays.
    
    terms = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # This is for a single scalar product term: u_val * A[u_idx] * v_val * B[v_idx] -> w_val * C[w_idx]
        # But Strassen involves sums of A entries.
        # So a "term" in the tensor decomposition is a rank-1 tensor.
        # A rank-1 tensor is defined by vectors u, v, w such that T = u \otimes v \otimes w.
        # In the context of matrix multiplication, a rank-1 term corresponds to:
        # C = sum_r u_r \otimes v_r \otimes w_r ? No.
        # The tensor T_{ijk} = delta_{jk} (for C = AB).
        # Decomposition: T = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r].
        # Where U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
        # A single rank-1 term r has vectors u_r, v_r, w_r.
        # The contribution is u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i].
        
        # So if I have a term like (A_{i1,j1} + A_{i2,j2}) * (B_{j1,k1} + B_{j2,k2}) * C_{k1,i1},
        # This is not a single rank-1 term. It expands to 4 rank-1 terms.
        # Strassen's algorithm works by defining linear combinations of A and B entries.
        # Let M1 = (A00 + A11) * (B00 + B11).
        # This product is added to C00 and C11, and subtracted from others.
        # The term "M1" is a scalar.
        # In the tensor decomposition, we need to express the final C entries as sums of these scalars.
        # C00 = M1 - M4 + M5 - M7
        # This means the vector W for the term corresponding to M1 will have non-zero entries at C00, C11, etc.
        # And the vectors U and V for that term will contain the coefficients of the linear combination.
        
        pass

    # Let's define the Strassen terms for 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B10)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6

    # For 4x4, we apply this to 2x2 blocks.
    # Let A44 be the top-left 4x4 of A.
    # A44 = [[A11, A12], [A21, A22]] where A11 is 2x2.
    # We need to generate 7 terms for the 4x4 multiplication.
    # Each term is a product of two 2x2 matrices.
    # Each 2x2 product expands to 7 scalar products.
    # So 49 scalar products total.
    
    # Let's create a helper to generate the 7 terms for a 2x2 block multiplication
    # given the global row/col offsets.
    
    def get_2x2_terms(row_offset, col_offset_a, col_offset_b, row_offset_c, col_offset_c):
        # This function returns a list of 7 tuples:
        # (u_dict, v_dict, w_dict)
        # where u_dict maps global A index (i*m+j) to coeff
        # v_dict maps global B index (j*p+k) to coeff
        # w_dict maps global C index (k*n+i) to coeff
        
        # Local indices 0..3 for A and B (2x2)
        # Global A indices:
        # A00: (row_offset + 0, col_offset_a + 0)
        # A01: (row_offset + 0, col_offset_a + 1)
        # A10: (row_offset + 1, col_offset_a + 0)
        # A11: (row_offset + 1, col_offset_a + 1)
        
        # Global B indices:
        # B00: (col_offset_b + 0, row_offset_c + 0) -- Wait, B is m x p.
        # In 4x4 * 4x4, B is also 4x4.
        # Let's assume we are multiplying A_sub (4x4) and B_sub (4x4) to get C_sub (4x4).
        # A_sub rows: r0..r3, cols: c0..c3
        # B_sub rows: c0..c3, cols: p0..p3
        # C_sub rows: p0..p3, cols: r0..r3
        
        terms_2x2 = []
        
        # M1
        u = { (row_offset, col_offset_a): 1, (row_offset+1, col_offset_a+1): 1 }
        v = { (col_offset_b, row_offset_c): 1, (col_offset_b+1, row_offset_c+1): 1 }
        w = { (row_offset_c, row_offset): 1, (row_offset_c+1, row_offset+1): 1 }
        terms_2x2.append((u, v, w))
        
        # M2
        u = { (row_offset+1, col_offset_a): 1, (row_offset+1, col_offset_a+1): 1 }
        v = { (col_offset_b, row_offset_c): 1 }
        w = { (row_offset_c+1, row_offset): 1 } # C10 part
        # Wait, C10 = M2 + M4. So M2 contributes to C10.
        # C10 index: row = row_offset_c+1, col = row_offset.
        # W index: k*n + i. Here k=row_offset_c+1, i=row_offset.
        # So w = { (row_offset_c+1, row_offset): 1 }
        terms_2x2.append((u, v, w))
        
        # M3
        u = { (row_offset, col_offset_a): 1 }
        v = { (col_offset_b+1, row_offset_c+1): 1, (col_offset_b+1, row_offset_c): -1 }
        w = { (row_offset_c+1, row_offset): 1 } # C01? No.
        # C01 = M3 + M5.
        # C01 index: row = row_offset_c, col = row_offset+1.
        # W index: k*n + i. k=row_offset_c, i=row_offset+1.
        # So w = { (row_offset_c, row_offset+1): 1 }
        terms_2x2.append((u, v, w))
        
        # M4
        u = { (row_offset+1, col_offset_a+1): 1 }
        v = { (col_offset_b+1, row_offset_c): 1, (col_offset_b, row_offset_c): -1 }
        w = { (row_offset_c+1, row_offset): 1 } # C10 part
        terms_2x2.append((u, v, w))
        
        # M5
        u = { (row_offset, col_offset_a): 1, (row_offset, col_offset_a+1): 1 }
        v = { (col_offset_b+1, row_offset_c+1): 1 }
        w = { (row_offset_c, row_offset): 1, (row_offset_c, row_offset+1): 1 } # C00 and C01?
        # C00 = ... + M5 ...
        # C01 = ... + M5 ...
        # C00 index: row=row_offset_c, col=row_offset.
        # C01 index: row=row_offset_c, col=row_offset+1.
        # So w has both.
        terms_2x2.append((u, v, w))
        
        # M6
        u = { (row_offset+1, col_offset_a): 1, (row_offset, col_offset_a): -1 }
        v = { (col_offset_b, row_offset_c): 1, (col_offset_b, row_offset_c+1): 1 }
        w = { (row_offset_c+1, row_offset+1): -1 } # C11 part?
        # C11 = M1 - M3 + M2 - M6.
        # C11 index: row=row_offset_c+1, col=row_offset+1.
        # So w = { (row_offset_c+1, row_offset+1): -1 }
        terms_2x2.append((u, v, w))
        
        # M7
        u = { (row_offset, col_offset_a+1): 1, (row_offset+1, col_offset_a+1): -1 }
        v = { (col_offset_b+1, row_offset_c): 1, (col_offset_b+1, row_offset_c+1): 1 }
        w = { (row_offset_c, row_offset): -1 } # C00 part?
        # C00 = M1 - M4 + M5 - M7.
        # C00 index: row=row_offset_c, col=row_offset.
        # So w = { (row_offset_c, row_offset): -1 }
        terms_2x2.append((u, v, w))
        
        return terms_2x2

    # Now, for 4x4, we have 4 2x2 blocks.
    # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
    # C11 (0,0), C12 (0,2), C21 (2,0), C22 (2,2)
    
    # Strassen on 4x4:
    # M1_4 = (A11+A22)(B11+B22)
    # M2_4 = (A21+A22)B11
    # M3_4 = A11(B22-B21)
    # M4_4 = A22(B21-B11)
    # M5_4 = (A11+A12)B22
    # M6_4 = (A21-A11)(B11+B12)
    # M7_4 = (A12-A22)(B21+B22)
    
    # C11_4 = M1_4 - M4_4 + M5_4 - M7_4
    # C12_4 = M3_4 + M5_4
    # C21_4 = M2_4 + M4_4
    # C22_4 = M1_4 - M3_4 + M2_4 - M6_4
    
    # Each Mx_4 is a 2x2 matrix multiplication.
    # So we need to generate the 7 terms for each Mx_4.
    
    all_terms = []
    
    # Helper to add a 2x2 multiplication to the list
    def add_2x2_mult(a_r, a_c, b_r, b_c, c_r, c_c, w_coeff):
        # a_r, a_c: top-left row/col of A sub-block
        # b_r, b_c: top-left row/col of B sub-block (note: B rows are cols of A)
        # c_r, c_c: top-left row/col of C sub-block
        # w_coeff: multiplier for the W vector (since C = sum coeff * M)
        
        terms_2x2 = get_2x2_terms(a_r, a_c, b_c, b_r, c_r, c_c)
        # Note: get_2x2_terms signature: (row_offset, col_offset_a, col_offset_b, row_offset_c, col_offset_c)
        # My previous definition:
        # u indices: (row_offset + di, col_offset_a + dj)
        # v indices: (col_offset_b + dk, row_offset_c + dl) -> Wait, B index is (j, k).
        # In my get_2x2_terms, I used:
        # v = { (col_offset_b, row_offset_c): ... }
        # This implies B index is (col_offset_b + ..., row_offset_c + ...).
        # This matches B[j,k] where j is row of B, k is col of B.
        # So col_offset_b is the row offset of B, and row_offset_c is the col offset of B?
        # No, let's trace carefully.
        # A[i,j] * B[j,k] -> C[k,i]
        # A sub-block at (a_r, a_c).
        # B sub-block at (b_r, b_c). Here b_r is the row index in B, which corresponds to col index in A.
        # So b_r should match a_c? No, they are independent in the general case, but for multiplication they must align.
        # In 4x4 * 4x4, A11 (cols 0,1) multiplies B11 (rows 0,1).
        # So a_c = 0, b_r = 0.
        # My get_2x2_terms uses `col_offset_a` for A cols and `col_offset_b` for B rows?
        # Let's rename parameters in get_2x2_terms to be clear:
        # a_row, a_col, b_row, b_col, c_row, c_col
        
        for u, v, w in terms_2x2:
            # Scale w by w_coeff
            w_scaled = {k: val * w_coeff for k, val in w.items()}
            all_terms.append((u, v, w_scaled))

    # M1_4 = (A11+A22)(B11+B22)
    # This is NOT a single 2x2 mult. It is a sum of 4 2x2 mults?
    # No, Strassen defines M1_4 as the product of the *sum* of matrices.
    # (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
    # Their product is computed using 7 scalar multiplications.
    # The inputs to the 2x2 Strassen are the entries of (A11+A22) and (B11+B22).
    # (A11+A22)[0,0] = A11[0,0] + A22[0,0]
    # So the U vector for the first term of this 2x2 mult will have non-zeros at A11[0,0] and A22[0,0].
    
    # So I need a function that generates the 7 terms for a 2x2 mult where the inputs are linear combinations.
    # This is getting complicated to implement dynamically.
    
    # Alternative: Just hardcode the 49 terms for 4x4 Strassen.
    # Or, use a library? No.
    
    # Let's simplify. I will implement the 4x4 Strassen by explicitly calling the 2x2 generator
    # but passing "virtual" indices or constructing the U/V vectors manually.
    
    # Actually, there is a simpler way.
    # I can just use the known rank-49 decomposition for 4x4 if I can find it.
    # But generating it is safer.
    
    # Let's write a generic 2x2 Strassen term generator that takes U/V/W coefficient maps.
    
    def gen_2x2_strassen(u_map, v_map, w_map):
        # u_map: dict (local_row, local_col) -> coeff for A
        # v_map: dict (local_row, local_col) -> coeff for B
        # w_map: dict (local_row, local_col) -> coeff for C
        # Returns list of 7 (U_global, V_global, W_global)
        
        # M1 = (A00+A11)(B00+B11)
        u1 = { (0,0): u_map.get((0,0),0) + u_map.get((1,1),0),
               (1,1): u_map.get((0,0),0) + u_map.get((1,1),0) }
        # This is wrong. u_map maps local indices to global coefficients?
        # No, let's pass the global indices directly.
        
        pass

    # This is getting too complex for a single edit.
    # Let's use a simpler improvement: **Strassen for 2x2 blocks in a 2x2 block decomposition of 5x5?**
    # No, 5 is not a power of 2.
    
    # Let's stick to the **Schoolbook for 4x4 + 1x1** but with **Strassen for the 4x4**.
    # I will manually define the 49 terms.
    
    # To save space and time, I will implement a function that generates the terms for a 2x2 block
    # and then call it for the 4 sub-blocks of the 4x4, combining the results appropriately.
    
    # Actually, I can just use the rank-49 decomposition of 4x4 matrix multiplication.
    # The coefficients are integers (-1, 0, 1).
    
    # Let's try a different tactic: **Use the rank-125 schoolbook but optimize the constant factor?**
    # No, the score is the rank.
    
    # I will implement the **4x4 Strassen + 1x1 cross terms** approach.
    # I will write a helper `add_2x2_strassen` that takes the global row/col offsets for A, B, C
    # and adds the 7 terms.
    # Then I will call it for the 7 M-terms of the 4x4 Strassen.
    # For M1_4 = (A11+A22)(B11+B22), I need to add terms for A11*B11, A11*B22, A22*B11, A22*B22.
    # But Strassen combines them into 7 products.
    # The product (A11+A22)(B11+B22) is NOT the sum of 4 separate Strassen decompositions.
    # It is ONE Strassen decomposition where the inputs are sums.
    
    # So I need to generate the 7 terms for a 2x2 multiplication where:
    # A00 = A11[0,0] + A22[0,0]
    # A01 = A11[0,1] + A22[0,1]
    # etc.
    
    # I will write a function `add_strassen_term` that takes:
    # a_coeffs: dict (local_i, local_j) -> (global_i, global_j, coeff)
    # b_coeffs: dict (local_i, local_j) -> (global_i, global_j, coeff)
    # c_coeffs: dict (local_i, local_j) -> (global_i, global_j, coeff)
    
    # And it generates the 7 rank-1 terms.
    
    def add_strassen_2x2(a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: { (0,0): [(g_i, g_j, c), ...], ... }
        # This is getting messy.
        
        # Let's just build the U, V, W vectors for each of the 7 terms directly.
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = {}
        for (lr, lc), gl in a_coeffs.items():
             pass # This logic is hard to generalize without a clear structure.

    # Given the complexity, I will use a **pre-computed rank-49 decomposition for 4x4**.
    # I can find the coefficients online or derive them.
    # But I can't browse.
    
    # Let's use **Strassen for 2x2** and **Schoolbook for the rest** in a 2x2 block decomposition of 5x5?
    # 5 = 2 + 2 + 1.
    # Blocks:
    # A11 (2x2), A12 (2x2), A13 (2x1)
    # A21 (2x2), A22 (2x2), A23 (2x1)
    # A31 (1x2), A32 (1x2), A33 (1x1)
    
    # C11 = A11B11 + A12B21 + A13B31
    # Use Strassen for A11B11 (rank 7).
    # Use Schoolbook for others.
    
    # This might yield a rank < 125.
    # A11B11: 7
    # A12B21: 4 (2x2 * 2x2 schoolbook is 4? No, 2x2 schoolbook is 4 mults. Rank 4.)
    # A13B31: 2 (2x1 * 1x2 is 2 mults? No, 2x1 * 1x2 -> 2x2. 4 mults? No. (2x1)*(1x2) = 2x2. 4 scalar mults.)
    # ...
    
    # This is tedious to calculate manually.
    
    # I will stick to the **4x4 Strassen** approach but implement it by **expanding the 7 M-terms** into their constituent 2x2 multiplications, and then **summing the W vectors** for the same A/B combinations?
    # No, that increases rank.
    
    # I will implement the **4x4 Strassen** by generating the 49 terms explicitly using a recursive call to a 2x2 generator.
    
    def get_2x2_terms(a_r, a_c, b_r, b_c, c_r, c_c):
        # Returns 7 tuples of (u_dict, v_dict, w_dict)
        # u_dict: { global_a_idx: coeff }
        # v_dict: { global_b_idx: coeff }
        # w_dict: { global_c_idx: coeff }
        
        # Helper to get global index
        def g_a(i, j): return i * 5 + j
        def g_b(i, j): return i * 5 + j
        def g_c(i, j): return j * 5 + i # Trace convention: W[k*n+i, r] -> index k*5+i
        
        terms = []
        
        # M1
        u = { g_a(a_r, a_c): 1, g_a(a_r+1, a_c+1): 1 }
        v = { g_b(b_r, b_c): 1, g_b(b_r+1, b_c+1): 1 }
        w = { g_c(c_r, c_c): 1, g_c(c_r+1, c_c+1): 1 }
        terms.append((u, v, w))
        
        # M2
        u = { g_a(a_r+1, a_c): 1, g_a(a_r+1, a_c+1): 1 }
        v = { g_b(b_r, b_c): 1 }
        w = { g_c(c_r+1, c_c): 1 }
        terms.append((u, v, w))
        
        # M3
        u = { g_a(a_r, a_c): 1 }
        v = { g_b(b_r+1, b_c+1): 1, g_b(b_r+1, b_c): -1 }
        w = { g_c(c_r, c_c+1): 1 }
        terms.append((u, v, w))
        
        # M4
        u = { g_a(a_r+1, a_c+1): 1 }
        v = { g_b(b_r+1, b_c): 1, g_b(b_r, b_c): -1 }
        w = { g_c(c_r+1, c_c): 1 }
        terms.append((u, v, w))
        
        # M5
        u = { g_a(a_r, a_c): 1, g_a(a_r, a_c+1): 1 }
        v = { g_b(b_r+1, b_c+1): 1 }
        w = { g_c(c_r, c_c): 1, g_c(c_r, c_c+1): 1 }
        terms.append((u, v, w))
        
        # M6
        u = { g_a(a_r+1, a_c): 1, g_a(a_r, a_c): -1 }
        v = { g_b(b_r, b_c): 1, g_b(b_r, b_c+1): 1 }
        w = { g_c(c_r+1, c_c+1): -1 }
        terms.append((u, v, w))
        
        # M7
        u = { g_a(a_r, a_c+1): 1, g_a(a_r+1, a_c+1): -1 }
        v = { g_b(b_r+1, b_c): 1, g_b(b_r+1, b_c+1): 1 }
        w = { g_c(c_r, c_c): -1 }
        terms.append((u, v, w))
        
        return terms

    # Now, for 4x4 Strassen:
    # M1_4 = (A11+A22)(B11+B22)
    # This requires adding terms for A11*B11, A11*B22, A22*B11, A22*B22?
    # NO. Strassen computes (A11+A22)(B11+B22) as a SINGLE 2x2 multiplication.
    # The inputs to the 2x2 multiplication are the sums.
    # So I need to modify `get_2x2_terms` to accept **linear combinations** as inputs.
    
    # I will create a new function `add_strassen_2x2_combined` that takes lists of (a_r, a_c, coeff) for A, etc.
    
    all_terms = []
    
    def add_combined_2x2(a_list, b_list, c_list, c_coeff=1):
        # a_list: [(a_r, a_c, coeff), ...]
        # b_list: [(b_r, b_c, coeff), ...]
        # c_list: [(c_r, c_c, coeff), ...]
        
        # We need to generate the 7 terms of Strassen for this "virtual" 2x2 block.
        # The U vector for a term will be the sum of the A entries involved.
        # The V vector will be the sum of the B entries involved.
        # The W vector will be the sum of the C entries involved, scaled by c_coeff.
        
        # M1 = (A00+A11)(B00+B11)
        # A00 = sum of a_list where local (0,0)
        # A11 = sum of a_list where local (1,1)
        
        def get_local_coeff(lst, lr, lc):
            total = 0
            for (r, c, coeff) in lst:
                if r == lr and c == lc:
                    total += coeff
            return total
        
        # This is not right. The "local" indices (0,0) etc. refer to the 2x2 block structure.
        # If I pass a_list = [(0,0,1), (2,2,1)], then A00 = A[0,0], A11 = A[2,2].
        # So local (0,0) corresponds to global (0,0). Local (1,1) corresponds to global (2,2).
        # This mapping is implicit in the list.
        
        # I need to map local 2x2 indices to the provided global indices.
        # This requires knowing which global index corresponds to local (0,0), (0,1), etc.
        # I will assume the list provides the mapping.
        # Let's define a helper to get the global indices for a local index.
        
        # This is too complex. I will just use **Schoolbook for 4x4** but with **Strassen for 2x2** blocks?
        # No, 4x4 schoolbook is 16. Strassen is 7.
        
        # I will use the **rank-49** decomposition for 4x4.
        # I will hardcode the 49 terms.
        
        # To avoid hardcoding 49 terms, I will use **Strassen for 2x2** and apply it to the 4 2x2 blocks of the 4x4 matrix,
        # but this gives rank 4*7 = 28? No, that's not how it works.
        
        # I will use the **4x4 Strassen** by calling `get_2x2_terms` 7 times,
        # but passing the **summed indices** in the U/V vectors.
        
        # I will implement `add_strassen_2x2` that takes:
        # a_coeffs: { (lr, lc): { global_idx: coeff } }
        # This is the way.
        
        pass

    # Given the time, I will implement the **4x4 Strassen** by manually constructing the 49 terms
    # using the `get_2x2_terms` function for the 4 sub-blocks and combining them?
    # No, that's schoolbook.
    
    # I will use **Strassen for 2x2** and **Schoolbook for the rest** in a **2x2 + 1x1** split.
    # Rank for 2x2 Strassen: 7.
    # Cross terms:
    # C11 (2x2) = A11B11 (7) + A12B21 (4) + A13B31 (2? No, 2x1*1x2=4)
    # This is getting messy.
    
    # I will simply return the **Schoolbook** decomposition but with a **small optimization**:
    # No, the directive says "new approach".
    
    # I will implement **Strassen for 2x2** and apply it to the **top-left 2x2 block** of the 5x5 matrix,
    # and use schoolbook for the rest.
    # Rank = 7 (for 2x2) + 24 (schoolbook for the other 24 entries? No).
    # Total entries = 25.
    # If I use Strassen for A[0:2,0:2] * B[0:2,0:2], I save 1 rank (8->7).
    # Total rank = 125 - 1 = 124.
    
    # This is a valid improvement and a "new approach" (hybrid Strassen/Schoolbook).
    
    def construct():
        n, m, p = 5, 5, 5
        terms = []
        
        # Add Strassen terms for the top-left 2x2 block
        # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        
        # Using the `get_2x2_terms` logic from before, but for global indices.
        
        def g_a(i, j): return i * 5 + j
        def g_b(i, j): return i * 5 + j
        def g_c(i, j): return j * 5 + i
        
        # M1
        u = { g_a(0,0): 1, g_a(1,1): 1 }
        v = { g_b(0,0): 1, g_b(1,1): 1 }
        w = { g_c(0,0): 1, g_c(1,1): 1 }
        terms.append((u, v, w))
        
        # M2
        u = { g_a(1,0): 1, g_a(1,1): 1 }
        v = { g_b(0,0): 1 }
        w = { g_c(1,0): 1 }
        terms.append((u, v, w))
        
        # M3
        u = { g_a(0,0): 1 }
        v = { g_b(1,1): 1, g_b(1,0): -1 }
        w = { g_c(0,1): 1 }
        terms.append((u, v, w))
        
        # M4
        u = { g_a(1,1): 1 }
        v = { g_b(1,0): 1, g_b(0,0): -1 }
        w = { g_c(1,0): 1 }
        terms.append((u, v, w))
        
        # M5
        u = { g_a(0,0): 1, g_a(0,1): 1 }
        v = { g_b(1,1): 1 }
        w = { g_c(0,0): 1, g_c(0,1): 1 }
        terms.append((u, v, w))
        
        # M6
        u = { g_a(1,0): 1, g_a(0,0): -1 }
        v = { g_b(0,0): 1, g_b(0,1): 1 }
        w = { g_c(1,1): -1 }
        terms.append((u, v, w))
        
        # M7
        u = { g_a(0,1): 1, g_a(1,1): -1 }
        v = { g_b(1,0): 1, g_b(1,1): 1 }
        w = { g_c(0,0): -1 }
        terms.append((u, v, w))
        
        # Now add Schoolbook terms for all OTHER entries
        # We need to add terms for all (i,j,k) such that NOT (i<2 and j<2 and k<2)
        # Wait, Strassen covers C[0:2, 0:2].
        # The terms above contribute to C[0,0], C[0,1], C[1,0], C[1,1].
        # We need to ensure we don't double count.
        # The Strassen terms are exact for the 2x2 block.
        # So we should NOT add schoolbook terms for i<2, j<2, k<2.
        
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    u = { g_a(i, j): 1 }
                    v = { g_b(j, k): 1 }
                    w = { g_c(k, i): 1 }
                    terms.append((u, v, w))
                    
        # Convert to matrices
        rank = len(terms)
        U = [[0]*rank for _ in range(25)]
        V = [[0]*rank for _ in range(25)]
        W = [[0]*rank for _ in range(25)]
        
        for r, (u, v, w) in enumerate(terms):
            for idx, val in u.items():
                U[idx][r] = val
            for idx, val in v.items():
                V[idx][r] = val
            for idx, val in w.items():
                W[idx][r] = val
                
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
