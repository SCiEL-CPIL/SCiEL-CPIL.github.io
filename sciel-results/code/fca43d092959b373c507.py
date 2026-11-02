DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    # A is n x m = 2x2, B is m x p = 2x10, C is n x p = 2x10
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Strassen: treat B as 2x2 over 1x5 vectors
    # B[j, k] for j in {0,1}, k in {0..9}
    # Split k into two groups of 5: k in [0,4] and k in [5,9]
    
    # A entries: A[i,j] indexed as i*m + j = i*2 + j
    # B entries: B[j,k] indexed as j*p + k = j*10 + k
    # C entries: C[i,k] indexed as k*n + i = k*2 + i
    
    # 7 Strassen terms, each producing a 1x5 vector
    # For each term, for each of 5 positions, one rank-1 factor
    
    terms = []
    
    # M1 = (A00 + A11) * (B00 + B11)
    # B00 = B[0, 0:5], B11 = B[1, 5:10]
    terms.append({
        'A': [(0, 1), (3, 1)],  # A00, A11 with coeff 1
        'B_blocks': [(0, 0, 5, 1), (1, 5, 10, 1)]  # (row, start, end, coeff)
    })
    
    # M2 = (A10 + A11) * B00
    # B00 = B[0, 0:5]
    terms.append({
        'A': [(2, 1), (3, 1)],  # A10, A11
        'B_blocks': [(0, 0, 5, 1)]
    })
    
    # M3 = A00 * (B01 - B11)
    # B01 = B[0, 5:10], B11 = B[1, 5:10]
    terms.append({
        'A': [(0, 1)],
        'B_blocks': [(0, 5, 10, 1), (1, 5, 10, -1)]
    })
    
    # M4 = A11 * (B10 - B00)
    # B10 = B[1, 0:5], B00 = B[0, 0:5]
    terms.append({
        'A': [(3, 1)],
        'B_blocks': [(1, 0, 5, 1), (0, 0, 5, -1)]
    })
    
    # M5 = (A00 + A01) * B11
    # B11 = B[1, 5:10]
    terms.append({
        'A': [(0, 1), (1, 1)],
        'B_blocks': [(1, 5, 10, 1)]
    })
    
    # M6 = (A10 - A00) * (B00 + B01)
    # B00 = B[0, 0:5], B01 = B[0, 5:10]
    terms.append({
        'A': [(2, 1), (0, -1)],
        'B_blocks': [(0, 0, 5, 1), (0, 5, 10, 1)]
    })
    
    # M7 = (A01 - A11) * (B10 + B11)
    # B10 = B[1, 0:5], B11 = B[1, 5:10]
    terms.append({
        'A': [(1, 1), (3, -1)],
        'B_blocks': [(1, 0, 5, 1), (1, 5, 10, 1)]
    })
    
    # Now build U, V, W
    # Each term contributes 5 rank-1 factors (one per output position in the 1x5 vector)
    # For a term with A-combo and B-blocks, and for output position k (0-4):
    #   The B entry is B[j, start+k] for each B_block (j, start, end, coeff)
    #   The contribution to C[i, k] or C[i, 5+k] depends on which block
    
    # Wait, I need to be more careful about how the 1x5 vector maps to C.
    
    # Let me reconsider. The output C is 2x10. 
    # C[i, k] for i in {0,1}, k in {0..9}
    # Indexed as k*n + i = k*2 + i
    
    # In Strassen, each M_i is a 1x5 vector. But which rows of C does it contribute to?
    
    # Actually, I think I need to reconsider the structure. Let me re-derive.
    
    # C = A * B where A is 2x2, B is 2x10
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Using Strassen on the 2x2 "block" structure where B's columns are grouped:
    # Let B_L = B[:, 0:5] (2x5), B_R = B[:, 5:10] (2x5)
    # C_L = A * B_L (2x5), C_R = A * B_R (2x5)
    # C = [C_L | C_R]
    
    # For C_L = A * B_L, this is 2x2 times 2x5.
    # We can apply Strassen-like decomposition if we view B_L as 2x2 over 1x... no, B_L is 2x5.
    
    # Hmm, this is getting complicated. Let me just directly build the rank-35 decomposition.
    
    # Alternative approach: directly enumerate the 35 rank-1 terms.
    
    # Each rank-1 term corresponds to one of the 7 Strassen products and one of the 5 positions.
    
    # For Strassen product s (0-6) and position t (0-4):
    #   U: linear combo of A entries
    #   V: linear combo of B entries (for the specific column t within the 5-column block)
    #   W: contributes to specific C entries
    
    # Let me just build it explicitly.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    col = 0
    for term in terms:
        A_combo = term['A']  # list of (A_index, coeff)
        B_blocks = term['B_blocks']  # list of (B_row, start, end, coeff)
        
        for t in range(5):
            # For this Strassen product and position t
            # U part: A entries
            for a_idx, a_coeff in A_combo:
                U[a_idx][col] = a_coeff
            
            # V part: B entries
            for b_row, b_start, b_end, b_coeff in B_blocks:
                b_col = b_start + t
                V[b_row * p + b_col][col] = b_coeff
            
            # W part: which C entries does this contribute to?
            # This is the tricky part. The Strassen product M_s contributes to
            # specific entries of C with specific coefficients.
            
            # I need to figure out the W coefficients based on the Strassen formula.
            # This requires knowing which Strassen term we're in and how it maps to C.
            
            col += 1
    
    # This approach is getting messy. Let me take a different, more direct approach.
    
    # Let me just verify: is rank 35 actually achievable?
    # The standard result is that the tensor rank of (2,2,10) is at most 35 via Strassen.
    # Let me construct it more carefully.
    
    # I'll restart with a cleaner construction.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # For each of the 7 Strassen products, and for each of 5 columns:
    # We need to determine the W coefficients.
    
    # Strassen products and their contributions to C:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1, C01 -= M1, C10 -= M1 (wait, let me recheck)
    
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # But here, each "Cij" is actually a 1x5 vector, and each "Mk" is a 1x5 vector.
    # So the contributions are vector-wise.
    
    # For the left half (columns 0-4):
    # C[0, 0:5] = M1_L + M4_L - M5_L + M7_L  where Mx_L is the left-half version
    # C[1, 0:5] = M2_L + M4_L
    # Wait, no. Let me re-derive for the specific block structure.
    
    # Actually, I realize the issue. When I split B into B_L (cols 0-4) and B_R (cols 5-9),
    # the Strassen decomposition applies to each half separately? No, that would give 7*2*5 = 70.
    
    # I think the correct interpretation is:
    # View the 2x10 B as a 2x2 matrix over the ring R = F^5 (5-dimensional vectors).
    # B[0,0] = B[0,0:5] (a 5-vector), B[0,1] = B[0,5:10] (a 5-vector)
    # B[1,0] = B[1,0:5] (a 5-vector), B[1,1] = B[1,5:10] (a 5-vector)
    
    # A is a 2x2 matrix over F (scalars).
    
    # C = A * B is 2x2 over R, so:
    # C[0,0] = A00*B[0,0] + A01*B[1,0] (5-vector)
    # C[0,1] = A00*B[0,1] + A01*B[1,1] (5-vector)
    # C[1,0] = A10*B[0,0] + A11*B[1,0] (5-vector)
    # C[1,1] = A10*B[0,1] + A11*B[1,1] (5-vector)
    
    # Strassen over the ring R:
    # M1 = (A00+A11) * (B[0,0]+B[1,1])  [product in R, which is componentwise for vectors]
    # M2 = (A10+A11) * B[0,0]
    # M3 = A00 * (B[0,1]-B[1,1])
    # M4 = A11 * (B[1,0]-B[0,0])
    # M5 = (A00+A01) * B[1,1]
    # M6 = (A10-A00) * (B[0,0]+B[0,1])
    # M7 = (A01-A11) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Each Mk is a 5-vector, computed as: (scalar combo of A) * (vector combo of B)
    # The multiplication in R is componentwise, so each component is a scalar*scalar product.
    
    # For each Mk and each component t (0-4):
    #   It's (sum of a_i * A[i]) * (sum of b_j * B[j, t])
    #   This is a bilinear form, decomposable as rank-1 in the tensor sense.
    
    # So for each of 7 Ms and 5 components, we get one rank-1 term. Total 35.
    
    # Now let's build U, V, W.
    
    # For Mk, component t:
    # U: coefficients on A entries
    # V: coefficients on B entries (specific column t within the 5-col block)
    # W: coefficients on C entries
    
    # The W coefficients depend on which C entries Mk contributes to.
    
    # Let me define for each Strassen term its contribution to C[0,0], C[0,1], C[1,0], C[1,1]
    
    strassen_contrib = [
        # (C00_coeff, C01_coeff, C10_coeff, C11_coeff)
        (1, 0, 0, 1),    # M1
        (0, 0, 1, -1),   # M2
        (0, 1, 0, 1),    # M3
        (1, 0, 1, 0),    # M4
        (-1, 1, 0, 0),   # M5
        (0, 0, 0, 1),    # M6
        (1, 0, 0, 0),    # M7
    ]
    
    # A coefficients for each Strassen term
    A_coeffs = [
        [(0, 1), (3, 1)],    # M1: A00 + A11
        [(2, 1), (3, 1)],    # M2: A10 + A11
        [(0, 1)],             # M3: A00
        [(3, 1)],             # M4: A11
        [(0, 1), (1, 1)],    # M5: A00 + A01
        [(2, 1), (0, -1)],   # M6: A10 - A00
        [(1, 1), (3, -1)],   # M7: A01 - A11
    ]
    
    # B block coefficients for each Strassen term
    # Each entry: (B_row, block_col, coeff) where block_col is 0 or 1 (left or right half)
    B_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1: B[0,0] + B[1,1]
        [(0, 0, 1)],               # M2: B[0,0]
        [(0, 1, 1), (1, 1, -1)],  # M3: B[0,1] - B[1,1]
        [(1, 0, 1), (0, 0, -1)],  # M4: B[1,0] - B[0,0]
        [(1, 1, 1)],               # M5: B[1,1]
        [(0, 0, 1), (0, 1, 1)],   # M6: B[0,0] + B[0,1]
        [(1, 0, 1), (1, 1, 1)],   # M7: B[1,0] + B[1,1]
    ]
    
    col = 0
    for s in range(7):
        a_coeff_list = A_coeffs[s]
        b_coeff_list = B_coeffs[s]
        contrib = strassen_contrib[s]
        
        for t in range(5):
            # U: A entries
            for a_idx, a_c in a_coeff_list:
                U[a_idx][col] = a_c
            
            # V: B entries
            for b_row, b_block, b_c in b_coeff_list:
                b_col = b_block * 5 + t  # actual column in B
                V[b_row * p + b_col][col] = b_c
            
            # W: C entries
            # C[0,0] corresponds to C[i=0, k=0:5], indexed as k*2+0 for k=0..4
            # C[0,1] corresponds to C[i=0, k=5:10], indexed as k*2+0 for k=5..9
            # C[1,0] corresponds to C[i=1, k=0:5], indexed as k*2+1 for k=0..4
            # C[1,1] corresponds to C[i=1, k=5:10], indexed as k*2+1 for k=5..9
            
            c00_c, c01_c, c10_c, c11_c = contrib
            if c00_c != 0:
                W[t * n + 0][col] = c00_c  # C[0, t] where t is 0-4, index = t*2+0
            if c01_c != 0:
                W[(5 + t) * n + 0][col] = c01_c  # C[0, 5+t], index = (5+t)*2+0
            if c10_c != 0:
                W[t * n + 1][col] = c10_c  # C[1, t], index = t*2+1
            if c11_c != 0:
                W[(5 + t) * n + 1][col] = c11_c  # C[1, 5+t], index = (5+t)*2+1
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
