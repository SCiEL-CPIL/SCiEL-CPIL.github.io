"""
Recursive Strassen-based decomposition for 4x4 matrix multiplication.
Rank 49 using recursive block structure.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication
    using recursive Strassen decomposition.
    
    The 4x4 matrices are partitioned into 2x2 blocks of 2x2 submatrices.
    We use Strassen's 7-multiplication algorithm at the block level,
    and standard 2x2 matrix multiplication (rank 8) at the sub-block level.
    Total rank: 7 * 8 = 56... wait, let me reconsider.
    
    Actually, for 4x4 = 2x2 blocks of 2x2:
    - Block-level: 7 products of 2x2 matrices
    - Each 2x2 product needs rank 8 (schoolbook for 2x2)
    - Total: 7 * 8 = 56
    
    But we can do better. Let me use a different approach.
    
    Actually, the known best upper bound for 4x4 matrix multiplication
    is 49, achieved by recursive application of Strassen's algorithm
    with a specific base case. Let me implement that properly.
    
    The key insight: instead of doing 2x2 schoolbook (rank 8), we can
    use a rank-7 decomposition for 2x2x2 over the reals... but that's
    Strassen again, giving 7*7=49.
    
    So: 4x4 = 2x2 blocks, each block product uses Strassen (7 mults),
    and each "mult" is a 2x2 matrix multiplication which itself uses
    Strassen (7 mults of scalars). Total: 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS
    
    # For 4x4 matrix multiplication with recursive Strassen:
    # We view 4x4 as 2x2 blocks of 2x2 matrices.
    # Level 1: 7 block products (each 2x2 x 2x2 -> 2x2)
    # Level 2: Each 2x2 product uses 7 scalar products
    # Total rank: 7 * 7 = 49
    
    # Define Strassen's 7 intermediate quantities for 2x2 matrices:
    # Given A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we have 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    #
    # Level 1 Strassen intermediates (each is 2x2 matrix product):
    # S1 = (A11 + A22)(B11 + B22)
    # S2 = (A21 + A22)B11
    # S3 = A11(B12 - B22)
    # S4 = A22(B21 - B11)
    # S5 = (A11 + A12)B22
    # S6 = (A21 - A11)(B11 + B12)
    # S7 = (A12 - A22)(B21 + B22)
    #
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # Each Si is a 2x2 matrix product, which we decompose using Strassen at level 2.
    
    # Let's define the decomposition more carefully.
    
    # U matrix: shape (16, 49) - corresponds to A[i,j] entries
    # V matrix: shape (16, 49) - corresponds to B[j,k] entries  
    # W matrix: shape (16, 49) - corresponds to C[k,i] entries (trace convention)
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen coefficients for level 1 (block level):
    # For each of the 7 Strassen terms, we need coefficients for A blocks and B blocks.
    # A block indices: 00, 01, 10, 11 -> row indices 0-3 in the 2x2 block structure
    # B block indices: 00, 01, 10, 11
    
    # Level 1: 7 Strassen terms
    # Each term involves sums/differences of A blocks and B blocks
    # For term t, A coefficients: dict mapping block (bi,bj) to coefficient
    # For term t, B coefficients: dict mapping block (bk,bl) to coefficient
    
    # S1 = (A11 + A22)(B11 + B22)
    # S2 = (A21 + A22)B11
    # S3 = A11(B12 - B22)
    # S4 = A22(B21 - B11)
    # S5 = (A11 + A12)B22
    # S6 = (A21 - A11)(B11 + B12)
    # S7 = (A12 - A22)(B21 + B22)
    
    # For each Strassen term, the A-side involves certain blocks with coefficients,
    # and the B-side involves certain blocks with coefficients.
    
    # Let's define for each level-1 term:
    # a_blocks: list of (block_row, block_col, coeff) for A
    # b_blocks: list of (block_row, block_col, coeff) for B
    
    level1_terms = [
        # S1 = (A11 + A22)(B11 + B22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # S2 = (A21 + A22)B11
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # S3 = A11(B12 - B22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # S4 = A22(B21 - B11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # S5 = (A11 + A12)B22
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # S6 = (A21 - A11)(B11 + B12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # S7 = (A12 - A22)(B21 + B22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # Level 2: For each 2x2 matrix product, use Strassen with 7 scalar products.
    # Given 2x2 matrices P and Q:
    # P = [[p11, p12], [p21, p22]], Q = [[q11, q12], [q21, q22]]
    # M1 = (p11 + p22)(q11 + q22)
    # M2 = (p21 + p22)q11
    # M3 = p11(q12 - q22)
    # M4 = p22(q21 - q11)
    # M5 = (p11 + p12)q22
    # M6 = (p21 - p11)(q11 + q12)
    # M7 = (p12 - p22)(q21 + q22)
    
    # For each level-2 term, P coefficients and Q coefficients for the 4 entries.
    level2_terms = [
        # M1 = (p11 + p22)(q11 + q22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (p21 + p22)q11
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3 = p11(q12 - q22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M4 = p22(q21 - q11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # M5 = (p11 + p12)q22
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6 = (p21 - p11)(q11 + q12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7 = (p12 - p22)(q21 + q22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # Now, for the full decomposition:
    # Each level-1 term t (7 total) is a product of 2x2 matrices.
    # That 2x2 product is decomposed into 7 level-2 scalar products.
    # So total 7 * 7 = 49 scalar products.
    
    # For each combination (t1, t2) where t1 is level-1 term index, t2 is level-2 term index:
    # The U vector gets contributions from A entries based on level-1 A blocks and level-2 P coefficients.
    # The V vector gets contributions from B entries based on level-1 B blocks and level-2 Q coefficients.
    # The W vector gets contributions to C entries based on how the result feeds into C blocks.
    
    # Let's think about this differently. 
    # 
    # The A matrix is 4x4. Block (bi, bj) of A consists of entries A[2*bi + si, 2*bj + sj] for si, sj in {0,1}.
    # Similarly for B and C.
    #
    # For level-1 term t1 with A-block coefficients a_blocks and B-block coefficients b_blocks:
    # The effective 2x2 matrix P has entries P[si, sj] = sum over (bi, bj, c) in a_blocks: c * A[2*bi+si, 2*bj+sj]
    # The effective 2x2 matrix Q has entries Q[si, sj] = sum over (bk, bl, c) in b_blocks: c * B[2*bk+si, 2*bl+sj]
    #
    # Wait, I need to be more careful. Let me redefine.
    #
    # For level-1 term t1:
    # P is a 2x2 matrix where P[si, sj] = sum_{(bi, bj, c) in a_blocks} c * A[2*bi+si, 2*bj+sj]
    # Q is a 2x2 matrix where Q[si, sj] = sum_{(bk, bl, c) in b_blocks} c * B[2*bk+si, 2*bl+sj]
    #
    # No wait, that's not quite right either. Let me reconsider.
    #
    # A is partitioned into 2x2 blocks. Block (r, c) of A is the 2x2 submatrix A[2r:2r+2, 2c:2c+2].
    # So block (r, c) has entries at positions (2r+i, 2c+j) for i,j in {0,1}.
    #
    # For level-1 term t1 with A-side specification:
    # P is formed as a linear combination of A blocks.
    # If a_blocks = [(r1, c1, coeff1), (r2, c2, coeff2), ...],
    # then P = coeff1 * A_block(r1, c1) + coeff2 * A_block(r2, c2) + ...
    # where A_block(r, c) is the 2x2 matrix at block position (r, c).
    #
    # So P[i, j] = sum_{(r, c, coeff) in a_blocks} coeff * A[2r+i, 2c+j] for i, j in {0,1}.
    #
    # Similarly for Q from B blocks.
    
    # Now for level-2 term t2:
    # M = P * Q (2x2 matrix product)
    # Using Strassen, M is computed via 7 scalar products.
    # For level-2 term t2 with P-coefficients p_coeffs and Q-coefficients q_coeffs:
    # p_coeffs specifies how to form a scalar from P: scalar_P = sum_{(i, j, c) in p_coeffs} c * P[i, j]
    # q_coeffs specifies how to form a scalar from Q: scalar_Q = sum_{(i, j, c) in q_coeffs} c * Q[i, j]
    # The product is scalar_P * scalar_Q.
    
    # Now, scalar_P = sum_{(i, j, c) in p_coeffs} c * P[i, j]
    #            = sum_{(i, j, c) in p_coeffs} c * sum_{(r, rc, cr) in a_blocks} cr * A[2r+i, 2rc+j]
    #            = sum_{(r, rc, cr) in a_blocks} sum_{(i, j, c) in p_coeffs} cr * c * A[2r+i, 2rc+j]
    
    # So for the U vector (which multiplies A entries):
    # For A entry at position (row, col) where row = 2r+i, col = 2rc+j:
    # U[row, col][col_index] = sum over all (r, rc, cr) in a_blocks where 2r+i = row and 2rc+j = col: cr
    #                         * sum over all (i', j', c') in p_coeffs where i' = i and j' = j: c'
    
    # This is getting complex. Let me just enumerate directly.
    
    col = 0
    for t1 in range(7):
        a_blocks, b_blocks = level1_terms[t1]
        for t2 in range(7):
            p_coeffs, q_coeffs = level2_terms[t2]
            
            # Build U vector for A entries
            # U[row][col] for row in 0..15 (4x4 matrix, row-major)
            for si in range(2):
                for sj in range(2):
                    # P[si, sj] = sum_{(r, rc, cr) in a_blocks} cr * A[2r+si, 2rc+sj]
                    p_coeff = 0
                    for (r, rc, cr) in a_blocks:
                        row = 2*r + si
                        col_a = 2*rc + sj
                        # Check if (si, sj) contributes to p_coeffs
                        pc = 0
                        for (pi, pj, pcval) in p_coeffs:
                            if pi == si and pj == sj:
                                pc += pcval
                        if pc != 0:
                            p_coeff += cr * pc
                    if p_coeff != 0:
                        row = 2*((a_blocks[0][0] if False else 0)) # This approach is flawed
                        # Let me redo this more carefully.
                    pass
            
            # Let me try a cleaner approach: for each A entry index, compute the coefficient.
            for a_row in range(4):
                for a_col in range(4):
                    # A entry at (a_row, a_col)
                    # Determine block: r = a_row // 2, si = a_row % 2, rc = a_col // 2, sj = a_col % 2
                    r = a_row // 2
                    si = a_row % 2
                    rc = a_col // 2
                    sj = a_col % 2
                    
                    coeff = 0
                    for (br, bc, bc_coeff) in a_blocks:
                        if br == r and bc == rc:
                            # This A entry is in block (r, rc) with block coefficient bc_coeff
                            # Now check if P[si, sj] is used in p_coeffs
                            for (pi, pj, pc_coeff) in p_coeffs:
                                if pi == si and pj == sj:
                                    coeff += bc_coeff * pc_coeff
                    U[a_row * 4 + a_col][col] = coeff
            
            # Build V vector for B entries
            for b_row in range(4):
                for b_col in range(4):
                    # B entry at (b_row, b_col)
                    br = b_row // 2
                    si = b_row % 2
                    bc = b_col // 2
                    sj = b_col % 2
                    
                    coeff = 0
                    for (bbr, bbc, bbc_coeff) in b_blocks:
                        if bbr == br and bbc == bc:
                            for (pi, pj, pc_coeff) in q_coeffs:
                                if pi == si and pj == sj:
                                    coeff += bbc_coeff * pc_coeff
                    V[b_row * 4 + b_col][col] = coeff
            
            # Build W vector for C entries
            # C = P * Q where P and Q are 2x2 matrices formed from A and B blocks.
            # C[si, sj] = sum_{t} P[si, t] * Q[t, sj]
            # Using Strassen, C is expressed in terms of the 7 M values.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # For level-2 term t2, its contribution to C entries:
            c_coeff = [0, 0, 0, 0]  # for C[0,0], C[0,1], C[1,0], C[1,1]
            if t2 == 0:  # M1
                c_coeff = [1, 0, 0, 1]
            elif t2 == 1:  # M2
                c_coeff = [0, 0, 1, -1]
            elif t2 == 2:  # M3
                c_coeff = [0, 1, 0, 1]
            elif t2 == 3:  # M4
                c_coeff = [1, 0, 1, 0]
            elif t2 == 4:  # M5
                c_coeff = [-1, 1, 0, 0]
            elif t2 == 5:  # M6
                c_coeff = [0, 0, 0, 1]
            elif t2 == 6:  # M7
                c_coeff = [1, 0, 0, 0]
            
            # The result M (2x2 matrix) contributes to C blocks based on level-1 combination.
            # For level-1 term t1, the resulting 2x2 matrix S contributes to C blocks.
            # C11 = S1 + S4 - S5 + S7
            # C12 = S3 + S5
            # C21 = S2 + S4
            # C22 = S1 - S2 + S3 + S6
            
            # So for level-1 term t1, S contributes to C blocks with certain coefficients.
            s_to_c = [0, 0, 0, 0]  # for C[0,0], C[0,1], C[1,0], C[1,1] in block space
            if t1 == 0:  # S1
                s_to_c = [1, 0, 0, 1]
            elif t1 == 1:  # S2
                s_to_c = [0, 0, 1, -1]
            elif t1 == 2:  # S3
                s_to_c = [0, 1, 0, 1]
            elif t1 == 3:  # S4
                s_to_c = [1, 0, 1, 0]
            elif t1 == 4:  # S5
                s_to_c = [-1, 1, 0, 0]
            elif t1 == 5:  # S6
                s_to_c = [0, 0, 0, 1]
            elif t1 == 6:  # S7
                s_to_c = [1, 0, 0, 0]
            
            # Now, the total W coefficient for C entry (c_row, c_col):
            # c_row = 2*bi + si, c_col = 2*ci + sj where bi, ci are block indices
            # The block (bi, ci) of C gets contribution s_to_c[bi*2+ci] * M[si, sj]
            # where M[si, sj] is the (si, sj) entry of the 2x2 result from level-2 term t2.
            
            for c_block_row in range(2):
                for c_block_col in range(2):
                    # Block (c_block_row, c_block_col) of C
                    block_coeff = s_to_c[c_block_row * 2 + c_block_col]
                    if block_coeff == 0:
                        continue
                    for si in range(2):
                        for sj in range(2):
                            # M[si, sj] contribution from level-2 term t2
                            m_coeff = c_coeff[si * 2 + sj]
                            if m_coeff == 0:
                                continue
                            total_coeff = block_coeff * m_coeff
                            c_row = 2 * c_block_row + si
                            c_col = 2 * c_block_col + sj
                            W[c_row * 4 + c_col][col] = total_coeff
            
            col += 1
    
    # Convert to integers where possible
    for mat in [U, V, W]:
        for row in mat:
            for idx in range(len(row)):
                if abs(row[idx] - round(row[idx])) < 1e-10:
                    row[idx] = int(round(row[idx]))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
