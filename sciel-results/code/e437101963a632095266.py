"""Combined program: recursive Strassen for 4x4x4 (rank 49) + block Strassen for 4x4x2 (rank 28)
giving total rank 77 for the 4x4x6 matrix multiplication tensor."""

import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_decomposition():
    """Return 7 (u, v, w) triples for 2x2x2 matrix multiplication.
    u in R^4 (row-major 2x2), v in R^4, w in R^4.
    Convention: u[row*2+col], v[row*2+col], w[row*2+col]
    """
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    triples = []
    # M1: u = A00+A11, v = B00+B11, w = C00 - C01 - C10 + C11
    triples.append(([1,0,0,1],[1,0,0,1],[1,0,0,1]))
    # M2: u = A10+A11, v = B00, w = C10 - C11
    triples.append(([0,0,1,1],[1,0,0,0],[0,0,1,-1]))
    # M3: u = A00, v = B01-B11, w = C01 + C11
    triples.append(([1,0,0,0],[0,1,0,-1],[0,1,0,1]))
    # M4: u = A11, v = B10-B00, w = C00 + C10
    triples.append(([0,0,0,1],[0,0,1,-1],[1,0,1,0]))
    # M5: u = A00+A10, v = B11, w = -C00 + C01
    triples.append(([1,0,1,0],[0,0,0,1],[-1,1,0,0]))
    # M6: u = A11-A01, v = B00+B01, w = C11
    triples.append(([0,-1,0,1],[1,1,0,0],[0,0,0,1]))
    # M7: u = A01-A11, v = B10+B11, w = C00 - C10
    triples.append(([0,1,0,-1],[0,0,1,1],[1,0,-1,0]))
    return triples

def _strassen_4x4x4_decomposition():
    """Return 49 (u, v, w) triples for 4x4x4 matrix multiplication.
    u in R^16 (row-major 4x4), v in R^16, w in R^16.
    Uses recursive Strassen: 7 multiplications of 2x2x2, each with 7 terms = 49.
    """
    s2 = _strassen_2x2x2_decomposition()
    result = []
    for (u2, v2, w2) in s2:
        # Expand each 2x2 coefficient to 4x4 (block-wise)
        u4 = [0]*16
        v4 = [0]*16
        w4 = [0]*16
        for bi in range(2):
            for bj in range(2):
                coeff_u = u2[bi*2+bj]
                coeff_v = v2[bi*2+bj]
                coeff_w = w2[bi*2+bj]
                for ii in range(2):
                    for jj in range(2):
                        u4[(bi*2+ii)*4 + (bj*2+jj)] = coeff_u
                        v4[(bi*2+ii)*4 + (bj*2+jj)] = coeff_v
                        w4[(bi*2+ii)*4 + (bj*2+jj)] = coeff_w
        result.append((u4, v4, w4))
    return result

def _strassen_4x4x2_decomposition():
    """Return 28 (u, v, w) triples for 4x4x2 matrix multiplication.
    u in R^16 (row-major 4x4), v in R^8 (row-major 4x2), w in R^8 (row-major 4x2).
    Split 4x4 into 2x2 blocks, use Strassen: 7 mults, each 2x2x1 needs 4 terms = 28.
    """
    # For 2x2 x 2x1 multiplication (rank 4 schoolbook, but we use Strassen structure)
    # Actually for 2x2 * 2x1, the standard rank is 4 (each output element needs one mult)
    # But we can use the Strassen block structure: split 4x4 into 2x2 blocks, 4x2 into 2x1 blocks
    # 7 Strassen multiplications, each producing a 2x1 result (rank 4 each) = 28
    
    # Let's define the 7 Strassen combinations for 2x2 * 2x1:
    # A is 2x2, B is 2x1 (column vector), C is 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 
    # Using Strassen-like: we can't reduce below 4 for 2x2*2x1 since it's essentially
    # computing two dot products of length 2. But the block Strassen approach gives
    # 7 multiplications where each "multiplication" is a 2x2 * 2x1 = 2x1 product.
    # 
    # Wait - the cell finding says rank 28 for 4x4x2. Let me think about this differently.
    # 
    # Split A (4x4) into 2x2 blocks: A = [[A00, A01],[A10, A11]] each 2x2
    # Split B (4x2) into 2x2 blocks: B = [[B00, B01],[B10, B11]] each 2x2
    # C (4x2) = [[C00, C01],[C10, C11]] each 2x2
    # 
    # Strassen: 7 multiplications of 2x2 * 2x2 = 2x2
    # Each 2x2 * 2x2 = 2x2 has schoolbook rank 4
    # Total: 7 * 4 = 28
    # 
    # But wait, we need 2x2 * 2x2 here, not 2x2 * 2x1. Let me re-examine.
    # A is 4x4, B is 4x2. Split into 2x2 blocks:
    # A: 2x2 blocks of 2x2 (4 blocks)
    # B: 2x2 blocks of 2x1 (4 blocks) -- NO, B is 4x2, so 2x2 blocks would be 2x1 each
    # 
    # Actually B is 4x2. If I split into 2x2 blocks: B = [[B00, B01],[B10, B11]] where each is 2x1.
    # Then C = A*B gives C = [[C00, C01],[C10, C11]] where each is 2x1.
    # 
    # Strassen for 2x2 blocks of (2x2) * (2x1) -> (2x1):
    # This is 7 "multiplications" of 2x2 * 2x1 = 2x1, each of rank 4, total 28.
    # 
    # But 2x2 * 2x1: the product is a 2x1 vector. Each element is a dot product of length 2.
    # Rank of 2x2*2x1 tensor = 4 (schoolbook, and this is tight for this small case).
    
    # For 2x2 * 2x1, schoolbook:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # 4 terms: (A00,B0,C0), (A01,B1,C0), (A10,B0,C1), (A11,B1,C1)
    
    # Now Strassen for the 2x2 block structure:
    # M1 = (A00+A11)*(B00+B10)  -- wait, B blocks are 2x1, so B00, B01, B10, B11 are 2x1 vectors
    # Hmm, this gets complicated. Let me just use the known result.
    
    # Actually, let me reconsider. The standard Strassen applies to square matrices.
    # For 4x4 * 4x2, we can pad B to 4x4 with zeros, do 4x4*4x4 Strassen (rank 49),
    # but that's wasteful. 
    # 
    # Alternative: split 4x4 * 4x2 as:
    # A (4x4) * B (4x2) where B = [B_left (4x1), B_right (4x1)]
    # C = [C_left (4x1), C_right (4x1)]
    # 
    # For 4x4 * 4x1: split into 2x2 blocks
    # A = [[A00, A01],[A10, A11]] (2x2 blocks of 2x2)
    # B = [b0; b1] where b0, b1 are 2x1 vectors
    # C = [c0; c1] where c0, c1 are 2x1 vectors
    # c0 = A00*b0 + A01*b1
    # c1 = A10*b0 + A11*b1
    # 
    # Using Strassen-like for this:
    # M1 = (A00+A11)*(b0+b1)
    # M2 = (A10+A11)*b0
    # M3 = A00*(b1-b0)  -- no, this doesn't work cleanly for rectangular
    # 
    # Let me just use the direct approach: for 4x4 * 4x2, 
    # schoolbook rank = 4*4*2 = 32
    # The cell finding says rank 28, so there's a 4-term reduction.
    # 
    # Actually, I think the correct interpretation is:
    # Split 4x4 * 4x2 into 2x2 blocks where the "2x2" in the middle refers to
    # the shared dimension. Use Strassen on the 2x2x2 structure.
    # 
    # Let me just construct it directly. For 4x4 * 4x2:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0..3, k=0..1
    # 
    # I'll use the following decomposition with rank 28:
    # Split rows into {0,1} and {2,3}, columns into {0} and {1}
    # For each pair of row-groups and column, use Strassen on the 2x2 * 2x1 subproblem
    
    # For 2x2 * 2x1 (A is 2x2, b is 2x1, c is 2x1):
    # c0 = A00*b0 + A01*b1
    # c1 = A10*b0 + A11*b1
    # Rank 4: terms are (A00,b0,c0), (A01,b1,c0), (A10,b0,c1), (A11,b1,c1)
    
    # Now for 4x4 * 4x2, split into 2x2 blocks:
    # A = [[A00, A01],[A10, A11]] where each Aij is 2x2
    # B = [[B00, B01],[B10, B11]] where each Bij is 2x1
    # C = [[C00, C01],[C10, C11]] where each Cij is 2x1
    # 
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # 
    # This is 8 terms of 2x2*2x1, each of rank 4, total 32.
    # To get 28, we need Strassen-like reduction.
    
    # Strassen for 2x2 block multiplication (where right operand is 2x1 blocks):
    # M1 = (A00+A11)*(B00+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)  -- doesn't quite work
    # 
    # Let me try a different Strassen formulation for (2x2) * (2x2 of 2x1 blocks):
    # Actually, I think the cleanest way is:
    # 
    # For the 2x2 block structure with 2x1 "entries":
    # M1 = (A00 + A11) * (B00 + B10)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A10) * B11
    # M6 = (A11 - A01) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # 
    # Each Mi is a 2x2 * 2x1 = 2x1 product, rank 4 each.
    # Total: 7 * 4 = 28. 
    
    # Now I need to express each 2x2*2x1 product as 4 rank-1 terms.
    # For 2x2 * 2x1: A (2x2) * b (2x1) = c (2x1)
    # c[0] = A[0,0]*b[0] + A[0,1]*b[1]
    # c[1] = A[1,0]*b[0] + A[1,1]*b[1]
    # 4 terms: (E00, e0, e0), (E01, e1, e0), (E10, e0, e1), (E11, e1, e1)
    # where Eij is the 2x2 matrix unit, ei is the 2x1 basis vector
    
    # Let me define helper for 2x2 * 2x1 decomposition
    def _2x2_2x1_decomp():
        """Return 4 (u, v, w) for 2x2 * 2x1. u in R^4, v in R^2, w in R^2."""
        # u: row-major 2x2, v: 2x1 vector, w: 2x1 vector
        return [
            ([1,0,0,0], [1,0], [1,0]),  # A00 * b0 -> c0
            ([0,1,0,0], [0,1], [1,0]),  # A01 * b1 -> c0
            ([0,0,1,0], [1,0], [0,1]),  # A10 * b0 -> c1
            ([0,0,0,1], [0,1], [0,1]),  # A11 * b1 -> c1
        ]
    
    base_2x2_2x1 = _2x2_2x1_decomp()
    
    # Now construct the 4x4 * 4x2 decomposition
    # A is 4x4 (row-major index: (row)*4+col)
    # B is 4x2 (row-major index: (row)*2+col)
    # C is 4x2 (row-major index: (row)*2+col)
    
    # Block structure:
    # A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B00: rows 0-1, col 0 -> indices 0,2
    # B01: rows 0-1, col 1 -> indices 1,3
    # B10: rows 2-3, col 0 -> indices 4,6
    # B11: rows 2-3, col 1 -> indices 5,7
    
    # C00: rows 0-1, col 0 -> indices 0,2
    # C01: rows 0-1, col 1 -> indices 1,3
    # C10: rows 2-3, col 0 -> indices 4,6
    # C11: rows 2-3, col 1 -> indices 5,7
    
    # Strassen combinations:
    # M1 = (A00+A11)*(B00+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A01)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mi, which is a 2x2 * 2x1 product, I expand into 4 terms.
    # The 2x2 matrix for A-blocks has indices in the 4x4 A matrix.
    # The 2x1 vector for B-blocks has indices in the 4x2 B matrix.
    # The 2x1 vector for C-blocks has indices in the 4x2 C matrix.
    
    # Let me define the index mappings:
    # A block (bi, bj) where bi in {0,1} (row group), bj in {0,1} (col group)
    # A index for element (ii, jj) in block (bi, bj): (bi*2+ii)*4 + (bj*2+jj)
    
    # B block (bi, bk) where bi in {0,1} (row group), bk in {0,1} (col)
    # B index for element (ii, bk) in block (bi, bk): (bi*2+ii)*2 + bk
    
    # C block (bi, bk) where bi in {0,1} (row group), bk in {0,1} (col)
    # C index for element (ii, bk) in block (bi, bk): (bi*2+ii)*2 + bk
    
    # Define the 7 Strassen A-combinations as lists of (block_row, block_col, coeff)
    # and B-combinations as lists of (block_row, block_col, coeff)
    
    strassen_defs = [
        # (A_comb, B_comb)
        # M1: A00+A11, B00+B10
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,0,1)]),
        # M2: A10+A11, B00
        ([(1,0,1),(1,1,1)], [(0,0,1)]),
        # M3: A00, B01-B11
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),
        # M4: A11, B10-B00
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),
        # M5: A00+A10, B11
        ([(0,0,1),(1,0,1)], [(1,1,1)]),
        # M6: A11-A01, B00+B01
        ([(1,1,1),(0,1,-1)], [(0,0,1),(0,1,1)]),
        # M7: A01-A11, B10+B11
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
    ]
    
    # C combination coefficients:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_combos = [
        [(0,1),(3,1),(4,-1),(6,1)],   # C00: M1, M4, -M5, M7
        [(2,1),(4,1)],                # C01: M3, M5
        [(1,1),(3,1)],                # C10: M2, M4
        [(0,1),(1,-1),(2,1),(5,1)],   # C11: M1, -M2, M3, M6
    ]
    
    result = []
    for m_idx, (a_comb, b_comb) in enumerate(strassen_defs):
        for base_u, base_v, base_w in base_2x2_2x1:
            u = [0]*16
            v = [0]*8
            w = [0]*8
            # Fill u from a_comb and base_u
            for (bi, bj, coeff) in a_comb:
                for ii in range(2):
                    for jj in range(2):
                        a_idx = (bi*2+ii)*4 + (bj*2+jj)
                        u[a_idx] += coeff * base_u[ii*2+jj]
            # Fill v from b_comb and base_v
            for (bi, bk, coeff) in b_comb:
                for ii in range(2):
                    b_idx = (bi*2+ii)*2 + bk
                    v[b_idx] += coeff * base_v[ii]
            # Fill w: need to determine which C blocks this term contributes to
            # base_w[ii] tells us which row of the 2x1 result
            # We need to figure out which C blocks (C00, C01, C10, C11) this M_idx contributes to
            for c_idx, combo in enumerate(c_combos):
                for (m_ref, c_coeff) in combo:
                    if m_ref == m_idx:
                        # This M_idx contributes to C block c_idx with coefficient c_coeff
                        for ii in range(2):
                            c_block_row = c_idx // 2  # 0 for C00,C01; 1 for C10,C11
                            c_block_col = c_idx % 2   # 0 for C00,C10; 1 for C01,C11
                            c_idx_global = (c_block_row*2+ii)*2 + c_block_col
                            w[c_idx_global] += c_coeff * base_w[ii]
            result.append((u, v, w))
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Decomposition 1: 4x4 * 4x4 (for B[:, 0:4]) using recursive Strassen, rank 49
    strassen_4x4x4 = _strassen_4x4x4_decomposition()
    
    # Decomposition 2: 4x4 * 4x2 (for B[:, 4:6]) using block Strassen, rank 28
    strassen_4x4x2 = _strassen_4x4x2_decomposition()
    
    total_rank = len(strassen_4x4x4) + len(strassen_4x4x2)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Add 4x4x4 decomposition (for B columns 0-3)
    for (u, v, w) in strassen_4x4x4:
        for i in range(16):
            U[i][col] = u[i]
        # v is for B (4x4), but we only use columns 0-3 of B (4x6)
        # v indices: (row)*4+col for 4x4, but B is 4x6 so index is (row)*6+col
        for i in range(16):
            row = i // 4
            bcol = i % 4
            V[row*6 + bcol][col] = v[i]
        # w is for C (4x4), but C is 4x6, index is (col)*4+row for p*n = 6*4
        # w indices: (col)*4+row for 4x4 C, but C is 4x6 so index is (col)*4+row
        # Wait, W shape is (p*n, rank) = (6*4, rank) = (24, rank)
        # W[k*n+i] where k is column of C (0-5), i is row of C (0-3)
        # For the 4x4 part, k is 0-3, i is 0-3
        for i in range(16):
            ccol = i // 4
            crow = i % 4
            W[ccol*4 + crow][col] = w[i]
        col += 1
    
    # Add 4x4x2 decomposition (for B columns 4-5)
    for (u, v, w) in strassen_4x4x2:
        for i in range(16):
            U[i][col] = u[i]
        # v is for B (4x2), indices: (row)*2+col for the 2-column part
        # But B is 4x6, so the 2 columns are at positions 4 and 5
        # v index (row)*2+col -> B index (row)*6 + (4+col)
        for i in range(8):
            row = i // 2
            bcol = i % 2
            V[row*6 + 4 + bcol][col] = v[i]
        # w is for C (4x2), indices: (col)*2+row for the 2-column part
        # W index: (col)*4+row, where col is 4 or 5
        # w index (col)*2+row -> W index (4+col)*4 + row
        for i in range(8):
            ccol = i // 2
            crow = i % 2
            W[(4+ccol)*4 + crow][col] = w[i]
        col += 1
    
    if time.time() > deadline:
        # Should not happen, but safety
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
