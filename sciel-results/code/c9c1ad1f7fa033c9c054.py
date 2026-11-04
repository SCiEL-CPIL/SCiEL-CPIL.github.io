"""Rewritten program for 4x4*4x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank for 4x4*4x6. We'll use a decomposition that achieves rank 96.
    # We can do:
    # - For the 4x4x4 part (columns 0-3 of B), use Strassen rank 49
    # - For the 4x4x2 part (columns 4-5 of B), use schoolbook rank 32
    # - For the 4x4x1 part (column 5... wait, p=6 means columns 0-5)
    # Actually p=6, so we have 6 columns. Let's split as 4+2.
    # 4x4x4: Strassen rank 49
    # 4x4x2: schoolbook 4*4*2 = 32
    # Total: 81
    
    # But wait, let me think about this more carefully.
    # The tensor is T[i,j,k] = sum_{a,b} A[i,a]*B[a,b]*delta... 
    # Actually the matrix multiplication tensor for A(n*m) * B(m*p) -> C(n*p)
    # T_{(i,j,k)} where i is row of A, j is col of A / row of B, k is col of B
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # So the tensor has entries T[(i,j), (j,k), (i,k)] = 1 for all i,j,k.
    # Total entries: n*m*p = 4*4*6 = 96.
    
    # For the 4x4x4 subproblem (k=0..3), we can use Strassen.
    # For the 4x4x2 subproblem (k=4..5), we use schoolbook.
    
    # Strassen's algorithm for 2x2*2x2 -> 2x2 has rank 7.
    # For 4x4*4x4, we tile into 2x2 blocks of 2x2 matrices.
    # This gives 7 products, each of which is 2x2*2x2, each rank 7.
    # Total: 7*7 = 49.
    
    # Let's build this carefully.
    # A is 4x4, B is 4x4 (for the first 4 columns), C is 4x4.
    # Tile A into 4 blocks: A00, A01, A10, A11 (each 2x2)
    # Tile B into 4 blocks: B00, B01, B10, B11 (each 2x2)
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pk is a 2x2*2x2 product, which has rank 7 (Strassen for 2x2).
    # So total rank for 4x4x4 is 7*7 = 49.
    
    # For the 4x4x2 part (columns 4 and 5 of B):
    # C04 = sum_j A[i,j]*B[j,4]
    # C05 = sum_j A[i,j]*B[j,5]
    # This is schoolbook: 4*4*2 = 32 rank.
    
    # Total rank: 49 + 32 = 81.
    
    rank = 81
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Helper: Strassen 2x2*2x2 decomposition
    # Given 2x2 matrices A2 (entries a00,a01,a10,a11) and B2 (entries b00,b01,b10,b11)
    # Returns 7 linear forms for A2, 7 for B2, and for each of 4 output positions,
    # the linear combination of the 7 products.
    
    # Strassen linear forms for A (2x2):
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    strassen_A = [
        [1,0,0,1],   # A00+A11
        [0,0,1,1],   # A10+A11
        [1,0,0,0],   # A00
        [0,0,0,1],   # A11
        [1,1,0,0],   # A00+A01
        [1,0,-1,0],  # A10-A00
        [0,1,0,-1],  # A01-A11
    ]
    strassen_B = [
        [1,0,0,1],   # B00+B11
        [1,0,0,0],   # B00
        [0,1,0,-1],  # B01-B11
        [0,0,1,-1],  # B10-B00
        [0,0,0,1],   # B11
        [1,1,0,0],   # B00+B01
        [0,0,1,1],   # B10+B11
    ]
    # Output: C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11
    strassen_C = [
        [1,0,0,0,0,0,1],   # P1+P4-P5+P7 for C00... wait
    ]
    # Let me redo: for each output position, which products with what signs
    # C00: P1, P4, -P5, P7 -> indices 0,3,-4,6 with coeff 1,1,-1,1
    # C01: P3, P5 -> indices 2,4 with coeff 1,1
    # C10: P2, P4 -> indices 1,3 with coeff 1,1
    # C11: P1, -P2, P3, P6 -> indices 0,1,2,5 with coeff 1,-1,1,1
    
    # For each of the 7 Strassen products at the 4x4 level, we need to decompose
    # the 2x2*2x2 sub-product using Strassen again.
    
    # The 4x4 matrix A has blocks:
    # A00: rows 0-1, cols 0-1 -> A indices (0,1,2,3) in flattened (row-major: 0,1,2,3)
    # A01: rows 0-1, cols 2-3 -> A indices (4,5,6,7)
    # A10: rows 2-3, cols 0-1 -> A indices (8,9,10,11)
    # A11: rows 2-3, cols 2-3 -> A indices (12,13,14,15)
    
    # Block to global A index mapping (flattened row-major 4x4):
    # A00: [0,1,2,3]
    # A01: [4,5,6,7]
    # A10: [8,9,10,11]
    # A11: [12,13,14,15]
    
    A_block_offsets = {
        '00': [0,1,2,3],
        '01': [4,5,6,7],
        '10': [8,9,10,11],
        '11': [12,13,14,15],
    }
    
    # Similarly for B (4x4 part, cols 0-3):
    B_block_offsets = {
        '00': [0,1,2,3],
        '01': [4,5,6,7],
        '10': [8,9,10,11],
        '11': [12,13,14,15],
    }
    
    # C output (4x4, rows 0-3, cols 0-3):
    C_block_offsets = {
        '00': [0,1,2,3],
        '01': [4,5,6,7],
        '10': [8,9,10,11],
        '11': [12,13,14,15],
    }
    
    # For each of the 7 outer Strassen products, we have a 2x2*2x2 sub-problem.
    # The outer product k corresponds to:
    # P1: (A00+A11)*(B00+B11)
    # P2: (A10+A11)*B00
    # P3: A00*(B01-B11)
    # P4: A11*(B10-B00)
    # P5: (A00+A01)*B11
    # P6: (A10-A00)*(B00+B01)
    # P7: (A01-A11)*(B10+B11)
    
    # For each outer product, the A-side is a linear combination of A blocks,
    # and the B-side is a linear combination of B blocks.
    # Each such linear combination, when decomposed via Strassen at the 2x2 level,
    # gives 7 inner products.
    
    # Let's define the outer Strassen terms as (A_coeffs, B_coeffs) where
    # A_coeffs maps block -> coefficient, same for B.
    
    outer_strassen = [
        ({'00':1, '11':1}, {'00':1, '11':1}),  # P1
        ({'10':1, '11':1}, {'00':1}),            # P2
        ({'00':1}, {'01':1, '11':-1}),           # P3
        ({'11':1}, {'10':1, '00':-1}),           # P4
        ({'00':1, '01':1}, {'11':1}),            # P5
        ({'10':1, '00':-1}, {'00':1, '01':1}),   # P6
        ({'01':1, '11':-1}, {'10':1, '11':1}),   # P7
    ]
    
    # For each outer term, for each of the 7 inner Strassen terms:
    # The U contribution: for each A block involved, scale the inner Strassen A form
    # by the outer coefficient and the block offset.
    
    for outer_idx, (a_coeffs, b_coeffs) in enumerate(outer_strassen):
        for inner_idx in range(7):
            # U: for each A block in a_coeffs
            for blk, coeff in a_coeffs.items():
                a_form = strassen_A[inner_idx]
                for local_idx, a_coeff in enumerate(a_form):
                    if a_coeff != 0:
                        global_idx = A_block_offsets[blk][local_idx]
                        U[global_idx][col] += coeff * a_coeff
            
            # V: for each B block in b_coeffs
            for blk, coeff in b_coeffs.items():
                b_form = strassen_B[inner_idx]
                for local_idx, b_coeff in enumerate(b_form):
                    if b_coeff != 0:
                        global_idx = B_block_offsets[blk][local_idx]
                        V[global_idx][col] += coeff * b_coeff
            
            # W: the output of this inner product contributes to certain C positions.
            # The inner product computes M = L*R where L is the 2x2 A-side, R is 2x2 B-side.
            # M has 4 entries: m00, m01, m10, m11.
            # Each m entry is a linear combination of the 7 Strassen products.
            
            # Which C positions does this outer term contribute to?
            # From Strassen:
            # C00 += P1 + P4 - P5 + P7
            # C01 += P3 + P5
            # C10 += P2 + P4
            # C11 += P1 - P2 + P3 + P6
            
            # So outer term outer_idx (0=P1, 1=P2, 2=P3, 3=P4, 4=P5, 5=P6, 6=P7)
            # contributes to C blocks with certain coefficients.
            
            c_contrib = {
                0: {'00':1, '11':1},   # P1 -> C00 (+1), C11 (+1)
                1: {'10':1, '11':-1},  # P2 -> C10 (+1), C11 (-1)
                2: {'01':1, '11':1},   # P3 -> C01 (+1), C11 (+1)
                3: {'00':1, '10':1},   # P4 -> C00 (+1), C10 (+1)
                4: {'00':-1, '01':1},  # P5 -> C00 (-1), C01 (+1)
                5: {'11':1},           # P6 -> C11 (+1)
                6: {'00':1},           # P7 -> C00 (+1)
            }
            
            # Wait, I need to double-check. Let me re-derive:
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # So:
            # P1: C00(+1), C11(+1)
            # P2: C10(+1), C11(-1)
            # P3: C01(+1), C11(+1)
            # P4: C00(+1), C10(+1)
            # P5: C00(-1), C01(+1)
            # P6: C11(+1)
            # P7: C00(+1)
            
            # That matches what I had.
            
            c_map = c_contrib[outer_idx]
            
            # For each C block this contributes to, the inner 2x2 result M
            # has 4 entries. The C block entry (ci,cj) gets the corresponding
            # M entry.
            # M entries: m00 (inner idx 0), m01 (inner idx 1), m10 (inner idx 2), m11 (inner idx 3)
            # The linear forms for each M entry in terms of the 7 Strassen products:
            # m00 = P1 + P4 - P5 + P7
            # m01 = P3 + P5
            # m10 = P2 + P4
            # m11 = P1 - P2 + P3 + P6
            
            m_forms = [
                [1,0,0,0,0,0,1],   # m00: P1+P4-P5+P7... wait no
            ]
            # Actually m00 uses products 0,3,4,6 with coeff 1,1,-1,1
            # m01 uses products 2,4 with coeff 1,1
            # m10 uses products 1,3 with coeff 1,1
            # m11 uses products 0,1,2,5 with coeff 1,-1,1,1
            
            m_forms = [
                [1,0,0,1,-1,0,1],   # m00
                [0,0,1,0,1,0,0],    # m01
                [0,1,0,1,0,0,0],    # m10
                [1,-1,1,0,0,1,0],   # m11
            ]
            
            # Hmm wait, I'm confusing myself. The inner Strassen product computes
            # M = L*R using 7 products Q1..Q7 (which are the inner_idx).
            # The output M entries are linear combinations of Q1..Q7.
            # But in my decomposition, each (outer_idx, inner_idx) pair is ONE rank-1 term.
            # The U and V for this term capture the linear forms for L and R.
            # The W should capture how this particular Q contributes to the final C.
            
            # Actually, I think I'm overcomplicating this. Let me reconsider.
            
            # The correct way: the full 4x4x4 tensor decomposition has 49 rank-1 terms.
            # Each term corresponds to a pair (outer_idx, inner_idx).
            # For term (o, i):
            #   U: linear form on A entries
            #   V: linear form on B entries
            #   W: linear form on C entries
            # such that sum over all terms of U*V*W = the multiplication tensor.
            
            # The construction: 
            # The 4x4 product is computed by 7 outer Strassen products.
            # Each outer product is a 2x2*2x2 product computed by 7 inner Strassen products.
            # So term (o,i) corresponds to the i-th inner product of the o-th outer product.
            
            # For the U side: the o-th outer product has A-side = sum of A blocks with coeffs.
            # The i-th inner Strassen product of that has A-form = strassen_A[i].
            # So U for term (o,i) = for each A block blk with coeff c_o,blk,
            #   add c_o,blk * strassen_A[i][local_idx] to the corresponding A entry.
            # This is what I have above.
            
            # Similarly for V.
            
            # For W: the i-th inner product computes Q_i = (A-form_i) * (B-form_i).
            # Q_i contributes to the 2x2 result M of the o-th outer product.
            # M contributes to the C blocks according to the outer Strassen formula.
            
            # So W for term (o,i) = sum over C blocks that the o-th outer product contributes to,
            #   of (coeff from outer formula) * (linear form of Q_i in the M entries) * (M entry to C block mapping).
            
            # The M entries are m00,m01,m10,m11. Each M entry maps to a C block entry.
            # For outer product o, the C block mapping is:
            #   e.g., for o=0 (P1): C00 gets m00, C11 gets m11.
            
            # And each m entry is a linear combination of Q1..Q7:
            #   m00 = Q1 + Q4 - Q5 + Q7
            #   m01 = Q3 + Q5
            #   m10 = Q2 + Q4
            #   m11 = Q1 - Q2 + Q3 + Q6
            
            # So for term (o,i), the coefficient of Q_i in each m entry is known.
            # And each m entry maps to specific C block entries.
            
            # Let me define: for outer product o, the mapping from m entry to C block position:
            # All outer products compute a 2x2 result M, and M is placed in C according to the block.
            # P1 contributes to C00 and C11: C00 += M00, C11 += M11
            # P2 contributes to C10 and C11: C10 += M00, C11 -= M11
            # P3 contributes to C01 and C11: C01 += M00, C11 += M11
            # P4 contributes to C00 and C10: C00 += M00, C10 += M10
            # P5 contributes to C00 and C01: C00 -= M00, C01 += M11
            # P6 contributes to C11: C11 += M00
            # P7 contributes to C00: C00 += M00
            
            # Wait, that's not right either. Let me be more careful.
            
            # C00 = P1 + P4 - P5 + P7, where each Pk is a 2x2 matrix.
            # So C00[0,0] = P1[0,0] + P4[0,0] - P5[0,0] + P7[0,0]
            # C00[0,1] = P1[0,1] + P4[0,1] - P5[0,1] + P7[0,1]
            # etc.
            
            # So for outer product o, the contribution to C block b is:
            #   C_block_b += coeff * M_o
            # where M_o is the 2x2 result of the o-th outer product.
            
            # The mapping from outer product o to C blocks:
            # o=0 (P1): C00 += M, C11 += M
            # o=1 (P2): C10 += M, C11 -= M
            # o=2 (P3): C01 += M, C11 += M
            # o=3 (P4): C00 += M, C10 += M
            # o=4 (P5): C00 -= M, C01 += M
            # o=5 (P6): C11 += M
            # o=6 (P7): C00 += M
            
            # And M is a 2x2 matrix with entries m00,m01,m10,m11.
            # Each m entry is a linear combo of Q1..Q7.
            
            # So for term (o,i), W is:
            #   For each C block b that o contributes to with coeff c:
            #     For each local position (r,s) in the 2x2 block:
            #       coeff_in_m = (linear form of Q_i in m_{r,s})
            #       W[global_C_index] += c * coeff_in_m
            
            # Let me code this up.
            
            # C block to global index (for the 4x4 C, rows 0-3, cols 0-3):
            # C00: rows 0-1, cols 0-1 -> [0,1,2,3]
            # C01: rows 0-1, cols 2-3 -> [4,5,6,7]
            # C10: rows 2-3, cols 0-1 -> [8,9,10,11]
            # C11: rows 2-3, cols 2-3 -> [12,13,14,15]
            
            # Outer product to C block mapping:
            outer_to_c = [
                {'00':1, '11':1},   # P1
                {'10':1, '11':-1},  # P2
                {'01':1, '11':1},   # P3
                {'00':1, '10':1},   # P4
                {'00':-1, '01':1},  # P5
                {'11':1},           # P6
                {'00':1},           # P7
            ]
            
            # m entry to local (r,s) position:
            # m00 -> (0,0), m01 -> (0,1), m10 -> (1,0), m11 -> (1,1)
            # m_forms[m_entry][i] = coefficient of Q_{i+1} in m_entry
            
            # For term (o, inner_idx), the coefficient of Q_{inner_idx+1} in each m entry:
            m_coeff = [
                [1,0,0,1,-1,0,1],   # m00
                [0,0,1,0,1,0,0],    # m01
                [0,1,0,1,0,0,0],    # m10
                [1,-1,1,0,0,1,0],   # m11
            ]
            
            for c_blk, c_coeff in outer_to_c[outer_idx].items():
                c_offsets = C_block_offsets[c_blk]
                for m_entry in range(4):
                    q_coeff = m_coeff[m_entry][inner_idx]
                    if q_coeff == 0:
                        continue
                    r = m_entry // 2
                    s = m_entry % 2
                    # Local (r,s) in the 2x2 block -> global C index
                    if c_blk == '00':
                        global_c = r + s * 2
                    elif c_blk == '01':
                        global_c = 4 + r + s * 2
                    elif c_blk == '10':
                        global_c = 8 + r + s * 2
                    elif c_blk == '11':
                        global_c = 12 + r + s * 2
                    # W index: k*n + i where k is col of C, i is row of C
                    # C is 4x4, flattened row-major: C[i,k] at index i*4+k
                    # So global_c is i*4+k, meaning i = global_c // 4, k = global_c % 4
                    i_row = global_c // 4
                    k_col = global_c % 4
                    W[k_col * n + i_row][col] += c_coeff * q_coeff
            
            col += 1
    
    # Now add the schoolbook part for columns 4 and 5 of B.
    # C[i,4] = sum_j A[i,j]*B[j,4]
    # C[i,5] = sum_j A[i,j]*B[j,5]
    for i in range(n):
        for j in range(m):
            for k in [4, 5]:
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
