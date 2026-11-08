import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = 4, 4, 6

    def strassen_2x2x2():
        # Returns list of (u, v, w) vectors of length 4 for 2x2x2 Strassen
        # A = [[a, b], [c, d]] indexed as [a, b, c, d]
        # B = [[e, f], [g, h]] indexed as [e, f, g, h]
        # C = [[C00, C01], [C10, C11]] indexed as [C00, C01, C10, C11]
        terms = []
        # M1 = (a+d)(e+h)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (c+d)e
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = a(f-h)
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = d(g-e)
        terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]))
        # M5 = (a+b)h
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (c-a)(e+f)
        terms.append(([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]))
        # M7 = (b-d)(g+h)
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return terms

    def schoolbook_2x2x1():
        # Returns list of (u, v, w) vectors for 2x2 * 2x1
        # A = [[a, b], [c, d]] indexed as [a, b, c, d]
        # B = [[e], [f]] indexed as [e, f]
        # C = [[g], [h]] indexed as [g, h]
        # g = a*e + b*f
        # h = c*e + d*f
        terms = []
        # a*e -> g
        terms.append(([1, 0, 0, 0], [1, 0], [1, 0]))
        # b*f -> g
        terms.append(([0, 1, 0, 0], [0, 1], [1, 0]))
        # c*e -> h
        terms.append(([0, 0, 1, 0], [1, 0], [0, 1]))
        # d*f -> h
        terms.append(([0, 0, 0, 1], [0, 1], [0, 1]))
        return terms

    def build_2x2x2x3():
        # Build decomposition for 2x2 * 2x3 -> 2x3
        # A is 2x2 (4 entries), B is 2x3 (6 entries), C is 2x3 (6 entries)
        # Split B into B1 (2x2) and B2 (2x1)
        # C = A * B1 + A * B2
        # Use Strassen for A * B1 (7 terms) and Schoolbook for A * B2 (4 terms)
        # Total 11 terms
        
        terms = []
        
        # Add Strassen terms for A * B1
        # B1 is columns 0,1 of B. So B1 entries are B[0,0], B[0,1], B[1,0], B[1,1]
        # In flattened B of length 6: B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx2, B[1,1]=idx3
        # C1 is columns 0,1 of C. C[0,0], C[0,1], C[1,0], C[1,1]
        # In flattened C of length 6: C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
        
        strassen_terms = strassen_2x2x2()
        for u, v, w in strassen_terms:
            # u is length 4 (for A)
            # v is length 4 (for B1). Map to v_full length 6.
            v_full = [0, 0, 0, 0, 0, 0]
            v_full[0] = v[0]
            v_full[1] = v[1]
            v_full[2] = v[2]
            v_full[3] = v[3]
            
            # w is length 4 (for C1). Map to w_full length 6.
            w_full = [0, 0, 0, 0, 0, 0]
            w_full[0] = w[0]
            w_full[1] = w[1]
            w_full[2] = w[2]
            w_full[3] = w[3]
            
            terms.append((u, v_full, w_full))
        
        # Add Schoolbook terms for A * B2
        # B2 is column 2 of B. B[0,2], B[1,2]
        # In flattened B of length 6: B[0,2]=idx4, B[1,2]=idx5
        # C2 is column 2 of C. C[0,2], C[1,2]
        # In flattened C of length 6: C[0,2]=idx4, C[1,2]=idx5
        
        schoolbook_terms = schoolbook_2x2x1()
        for u, v, w in schoolbook_terms:
            # u is length 4 (for A)
            # v is length 2 (for B2). Map to v_full length 6.
            v_full = [0, 0, 0, 0, 0, 0]
            v_full[4] = v[0]
            v_full[5] = v[1]
            
            # w is length 2 (for C2). Map to w_full length 6.
            w_full = [0, 0, 0, 0, 0, 0]
            w_full[4] = w[0]
            w_full[5] = w[1]
            
            terms.append((u, v_full, w_full))
            
        return terms

    def build_4x4x6():
        # Build decomposition for 4x4 * 4x6 -> 4x6
        # Use Strassen at the top level.
        # A (4x4) split into 4 blocks A11, A12, A21, A22 (2x2 each)
        # B (4x6) split into 4 blocks B11, B12, B21, B22 (2x3 each)
        # C (4x6) split into 4 blocks C11, C12, C21, C22 (2x3 each)
        
        # Strassen 4x4x6 terms (7 top-level terms)
        # Each top-level term corresponds to a product of linear combinations of A blocks
        # and linear combinations of B blocks.
        # The result is a 2x3 block which is computed using the 2x2x2x3 decomposition (11 terms).
        
        # Top-level Strassen formulas for 4x4 * 4x6:
        # Let A = [[A11, A12], [A21, A22]]
        # Let B = [[B11, B12], [B21, B22]]
        # Let C = [[C11, C12], [C21, C22]]
        
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M6
        # C22 = M1 - M2 + M3 + M6
        
        # We will iterate over the 7 Strassen terms.
        # For each term, we have:
        # A_part: a linear combination of the 4 A blocks (2x2 each).
        # B_part: a linear combination of the 4 B blocks (2x3 each).
        # C_part: a linear combination of the 4 C blocks (2x3 each).
        
        # The A_part is a 2x2 matrix. The B_part is a 2x3 matrix.
        # We use the 2x2x2x3 decomposition (11 terms) to compute A_part * B_part.
        # Each of the 11 terms contributes to the C_part blocks according to the w vector
        # of the 2x2x2x3 decomposition.
        
        # Wait, the w vector of the 2x2x2x3 decomposition is of length 6 (for the 2x3 result).
        # The 2x3 result is a block in C.
        # The C_part in the top-level Strassen formula is a linear combination of the 4 C blocks.
        # So we need to map the 2x3 result to the appropriate C blocks.
        
        # Let's define the top-level terms explicitly.
        # Each top-level term t has:
        # - a list of (coeff, block_idx) for A
        # - a list of (coeff, block_idx) for B
        # - a list of (coeff, block_idx) for C
        
        # Block indices:
        # A: 0=A11, 1=A12, 2=A21, 3=A22
        # B: 0=B11, 1=B12, 2=B21, 3=B22
        # C: 0=C11, 1=C12, 2=C21, 3=C22
        
        top_terms = []
        
        # M1 = (A11 + A22) * (B11 + B22)
        # C contribution: C11 += 1*M1, C22 += 1*M1
        top_terms.append((
            [(1, 0), (1, 3)],
            [(1, 0), (1, 3)],
            [(1, 0), (1, 3)]
        ))
        
        # M2 = (A21 + A22) * B11
        # C contribution: C21 += 1*M2, C22 += -1*M2
        top_terms.append((
            [(1, 2), (1, 3)],
            [(1, 0)],
            [(1, 2), (-1, 3)]
        ))
        
        # M3 = A11 * (B12 - B22)
        # C contribution: C12 += 1*M3, C22 += 1*M3
        top_terms.append((
            [(1, 0)],
            [(1, 1), (-1, 3)],
            [(1, 1), (1, 3)]
        ))
        
        # M4 = A22 * (B21 - B11)
        # C contribution: C11 += 1*M4
        top_terms.append((
            [(1, 3)],
            [(1, 2), (-1, 0)],
            [(1, 0)]
        ))
        
        # M5 = (A11 + A12) * B22
        # C contribution: C11 += -1*M5, C12 += 1*M5
        top_terms.append((
            [(1, 0), (1, 1)],
            [(1, 3)],
            [(-1, 0), (1, 1)]
        ))
        
        # M6 = (A21 - A11) * (B11 + B12)
        # C contribution: C21 += 1*M6, C22 += 1*M6
        top_terms.append((
            [(1, 2), (-1, 0)],
            [(1, 0), (1, 1)],
            [(1, 2), (1, 3)]
        ))
        
        # M7 = (A12 - A22) * (B21 + B22)
        # C contribution: C11 += 1*M7
        top_terms.append((
            [(1, 1), (-1, 3)],
            [(1, 2), (1, 3)],
            [(1, 0)]
        ))
        
        # Now, for each top-level term, we expand using the 2x2x2x3 decomposition.
        all_terms = []
        
        # Dimensions:
        # A total: 4x4 = 16 entries.
        # A block (2x2) is 4 entries.
        # A block 0 (A11) indices: 0,1,4,5
        # A block 1 (A12) indices: 2,3,6,7
        # A block 2 (A21) indices: 8,9,12,13
        # A block 3 (A22) indices: 10,11,14,15
        
        # B total: 4x6 = 24 entries.
        # B block (2x3) is 6 entries.
        # B block 0 (B11) indices: 0,1,2, 6,7,8
        # B block 1 (B12) indices: 3,4,5, 9,10,11
        # B block 2 (B21) indices: 12,13,14, 18,19,20
        # B block 3 (B22) indices: 15,16,17, 21,22,23
        
        # C total: 6x4 = 24 entries.
        # C block (2x3) is 6 entries.
        # Wait, the output convention is W[k*n+i, r].
        # C is n x p = 4 x 6.
        # W has shape (p*n, rank) = (24, rank).
        # W index is k*n + i = k*4 + i.
        # So C[i, k] is stored at W[k*4 + i].
        # Let's check the block structure in this flattened W space.
        
        # C block 0 (C11): i in {0,1}, k in {0,1}.
        # Indices: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 4; k=1,i=1 -> 5.
        # So C11 corresponds to W indices [0, 1, 4, 5].
        
        # C block 1 (C12): i in {0,1}, k in {2,3}.
        # Indices: k=2,i=0 -> 8; k=2,i=1 -> 9; k=3,i=0 -> 12; k=3,i=1 -> 13.
        # So C12 corresponds to W indices [8, 9, 12, 13].
        
        # C block 2 (C21): i in {2,3}, k in {0,1}.
        # Indices: k=0,i=2 -> 2; k=0,i=3 -> 3; k=1,i=2 -> 6; k=1,i=3 -> 7.
        # So C21 corresponds to W indices [2, 3, 6, 7].
        
        # C block 3 (C22): i in {2,3}, k in {2,3}.
        # Indices: k=2,i=2 -> 10; k=2,i=3 -> 11; k=3,i=2 -> 14; k=3,i=3 -> 15.
        # So C22 corresponds to W indices [10, 11, 14, 15].
        
        # The 2x2x2x3 decomposition produces a 2x3 matrix C_block.
        # Its entries are c00, c01, c02, c10, c11, c12 (row-major? or something else?)
        # The schoolbook/strassen definitions used row-major:
        # C = [[C00, C01, C02], [C10, C11, C12]]
        # So the 6 entries are [C00, C01, C02, C10, C11, C12].
        
        # We need to map these 6 entries to the W indices of the corresponding C block.
        
        # For C11 (i=0,1; k=0,1):
        # C00 -> C[0,0] -> W[0*4+0] = W[0]
        # C01 -> C[0,1] -> W[1*4+0] = W[4]
        # C02 -> N/A (C11 is 2x2? No, C11 is a block of C. C is 4x6. C11 is top-left 2x2? No.)
        
        # Wait, the block structure for C:
        # C is 4x6.
        # C11 is rows 0-1, cols 0-1. (2x2)
        # C12 is rows 0-1, cols 2-3. (2x2)
        # C21 is rows 2-3, cols 0-1. (2x2)
        # C22 is rows 2-3, cols 2-3. (2x2)
        
        # But B is 4x6.
        # B11 is rows 0-1, cols 0-1. (2x2)
        # B12 is rows 0-1, cols 2-3. (2x2)
        # B13 is rows 0-1, cols 4-5. (2x2)
        # B21 is rows 2-3, cols 0-1. (2x2)
        # B22 is rows 2-3, cols 2-3. (2x2)
        # B23 is rows 2-3, cols 4-5. (2x2)
        
        # My previous decomposition assumed B was split into 4 blocks of 2x3.
        # B1 = [[B11, B12], [B21, B22]] where each sub-block is 2x2? No.
        # If B is 4x6, and we split it into 4 blocks of 2x3:
        # B_top = B[0:2, :] (2x6)
        # B_bot = B[2:4, :] (2x6)
        # Then B_top is split into B11 (2x3) and B12 (2x3)?
        # No, the Strassen formula I used was for 4x4 * 4x4.
        # I need to adapt it for 4x4 * 4x6.
        
        # Let's restart the top-level structure for 4x4 * 4x6.
        # A (4x4) -> A11, A12, A21, A22 (2x2)
        # B (4x6) -> B1, B2 where B1 is 2x6 (top half), B2 is 2x6 (bottom half).
        # Then B1 -> B11 (2x3), B12 (2x3).
        # B2 -> B21 (2x3), B22 (2x3).
        
        # So the blocks are:
        # B11: B[0:2, 0:3]
        # B12: B[0:2, 3:6]
        # B21: B[2:4, 0:3]
        # B22: B[2:4, 3:6]
        
        # C (4x6) -> C1, C2 where C1 is 2x6, C2 is 2x6.
        # C1 -> C11 (2x3), C12 (2x3).
        # C2 -> C21 (2x3), C22 (2x3).
        
        # Strassen for 4x4 * 4x6:
        # M1 = (A11 + A22) * (B11 + B21)  ? No, B is split by rows.
        # Standard Strassen splits both matrices by rows and cols.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]] where B11 is 2x3, etc.
        
        # This matches my previous block definitions.
        # B11: B[0:2, 0:3]
        # B12: B[0:2, 3:6]
        # B21: B[2:4, 0:3]
        # B22: B[2:4, 3:6]
        
        # C11: C[0:2, 0:3]
        # C12: C[0:2, 3:6]
        # C21: C[2:4, 0:3]
        # C22: C[2:4, 3:6]
        
        # So the top-level Strassen formulas are:
        # M1 = (A11 + A22) * (B11 + B21)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M6
        # C22 = M1 - M2 + M3 + M6
        
        # This is exactly what I had before.
        # Now I just need to correctly map the 2x3 result of each inner term to the W indices.
        
        # Inner term result is a 2x3 matrix R.
        # R = [[r00, r01, r02], [r10, r11, r12]]
        
        # Mapping to W indices for each C block:
        
        # C11 (rows 0,1; cols 0,1,2):
        # r00 -> C[0,0] -> W[0*4+0] = 0
        # r01 -> C[0,1] -> W[1*4+0] = 4
        # r02 -> C[0,2] -> W[2*4+0] = 8
        # r10 -> C[1,0] -> W[0*4+1] = 1
        # r11 -> C[1,1] -> W[1*4+1] = 5
        # r12 -> C[1,2] -> W[2*4+1] = 9
        # Map: [0, 4, 8, 1, 5, 9]
        
        # C12 (rows 0,1; cols 3,4,5):
        # r00 -> C[0,3] -> W[3*4+0] = 12
        # r01 -> C[0,4] -> W[4*4+0] = 16
        # r02 -> C[0,5] -> W[5*4+0] = 20
        # r10 -> C[1,3] -> W[3*4+1] = 13
        # r11 -> C[1,4] -> W[4*4+1] = 17
        # r12 -> C[1,5] -> W[5*4+1] = 21
        # Map: [12, 16, 20, 13, 17, 21]
        
        # C21 (rows 2,3; cols 0,1,2):
        # r00 -> C[2,0] -> W[0*4+2] = 2
        # r01 -> C[2,1] -> W[1*4+2] = 6
        # r02 -> C[2,2] -> W[2*4+2] = 10
        # r10 -> C[3,0] -> W[0*4+3] = 3
        # r11 -> C[3,1] -> W[1*4+3] = 7
        # r12 -> C[3,2] -> W[2*4+3] = 11
        # Map: [2, 6, 10, 3, 7, 11]
        
        # C22 (rows 2,3; cols 3,4,5):
        # r00 -> C[2,3] -> W[3*4+2] = 14
        # r01 -> C[2,4] -> W[4*4+2] = 18
        # r02 -> C[2,5] -> W[5*4+2] = 22
        # r10 -> C[3,3] -> W[3*4+3] = 15
        # r11 -> C[3,4] -> W[4*4+3] = 19
        # r12 -> C[3,5] -> W[5*4+3] = 23
        # Map: [14, 18, 22, 15, 19, 23]
        
        w_maps = [
            [0, 4, 8, 1, 5, 9],   # C11
            [12, 16, 20, 13, 17, 21], # C12
            [2, 6, 10, 3, 7, 11], # C21
            [14, 18, 22, 15, 19, 23]  # C22
        ]
        
        # A block maps (to indices in U):
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        a_block_maps = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        
        # B block maps (to indices in V):
        # B is 4x6. Flattened: row-major.
        # B[i, j] -> i*6 + j
        # B11 (rows 0,1; cols 0,1,2):
        # 0,1,2, 6,7,8
        # B12 (rows 0,1; cols 3,4,5):
        # 3,4,5, 9,10,11
        # B21 (rows 2,3; cols 0,1,2):
        # 12,13,14, 18,19,20
        # B22 (rows 2,3; cols 3,4,5):
        # 15,16,17, 21,22,23
        b_block_maps = [
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5, 9, 10, 11],
            [12, 13, 14, 18, 19, 20],
            [15, 16, 17, 21, 22, 23]
        ]
        
        inner_terms = build_2x2x2x3()
        
        for a_coeffs, b_coeffs, c_coeffs in top_terms:
            # Construct u_full (length 16)
            u_full = [0] * 16
            for coeff, block_idx in a_coeffs:
                for k, u_idx in enumerate(a_block_maps[block_idx]):
                    u_full[u_idx] += coeff * 1 # Inner u is for the 2x2 block.
                    # Wait. The inner u is a vector of length 4 for the 2x2 block.
                    # The block map gives the global indices for the 4 entries of the block.
                    # So if inner u is [u0, u1, u2, u3], then:
                    # u_full[a_block_maps[block_idx][0]] += coeff * u0
                    # etc.
                    
            # We need to combine the top-level coefficients with the inner terms.
            # Let's do it properly.
            
            for u_inner, v_inner, w_inner in inner_terms:
                u_full = [0] * 16
                for coeff, block_idx in a_coeffs:
                    for k in range(4):
                        u_full[a_block_maps[block_idx][k]] += coeff * u_inner[k]
                
                v_full = [0] * 24
                for coeff, block_idx in b_coeffs:
                    for k in range(6):
                        v_full[b_block_maps[block_idx][k]] += coeff * v_inner[k]
                
                w_full = [0] * 24
                # w_inner is length 6.
                # We need to distribute it to the C blocks according to c_coeffs.
                # c_coeffs is a list of (coeff, block_idx).
                # But wait, the top-level formula says e.g. C11 += M1.
                # So the result of M1 (which is a 2x3 matrix) is added to C11.
                # So for each (coeff, block_idx) in c_coeffs:
                #   for k in range(6):
                #       w_full[w_maps[block_idx][k]] += coeff * w_inner[k]
                
                for coeff, block_idx in c_coeffs:
                    for k in range(6):
                        w_full[w_maps[block_idx][k]] += coeff * w_inner[k]
                
                all_terms.append((u_full, v_full, w_full))
        
        return all_terms

    terms = build_4x4x6()
    rank = len(terms)
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for t, (u, v, w) in enumerate(terms):
        U[:, t] = u
        V[:, t] = v
        W[:, t] = w
        
    # Verification
    # Check if U @ diag(1) @ V^T * ... reconstructs the tensor?
    # Actually, check if sum_t u_t \otimes v_t \otimes w_t = T
    # T[i, j, k] = 1 if A[i,j] B[j,k] contributes to C[i,k].
    # Let's just check a few random matrices.
    
    A_test = np.random.randint(0, 10, (4, 4))
    B_test = np.random.randint(0, 10, (4, 6))
    C_test = A_test @ B_test
    
    U_mat = np.array(U).T # (rank, 16)
    V_mat = np.array(V).T # (rank, 24)
    W_mat = np.array(W).T # (rank, 24)
    
    # C_calc[i, k] = sum_t (sum_j U[t, i*m+j] A[i,j]) (sum_l V[t, j*p+l] B[j,l]) W[t, k*n+i]
    # This is not quite right. The decomposition is:
    # C = sum_t (U_t^T A) (V_t^T B) W_t^T ? No.
    # C[i,k] = sum_t u_t[i] v_t[k] w_t ? No.
    # The tensor T is decomposed as sum_t u_t \otimes v_t \otimes w_t.
    # T[i, j, k] = sum_t u_t[i] v_t[j] w_t[k].
    # Then C = <T, A \otimes B> ? No.
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] ?
    # Actually, the standard definition is:
    # C = sum_t (U_t^T A) (B V_t) W_t^T? No.
    # Let's use the explicit sum.
    
    C_calc = np.zeros((4, 6))
    for t in range(rank):
        u = U[:, t]
        v = V[:, t]
        w = W[:, t]
        
        # u is length 16. Reshape to 4x4.
        U_t = u.reshape(4, 4)
        # v is length 24. Reshape to 4x6.
        V_t = v.reshape(4, 6)
        # w is length 24. Reshape to 6x4.
        W_t = w.reshape(6, 4)
        
        # Contribution to C is (U_t^T A) * (B V_t) * W_t^T ?
        # No, the product of linear forms is a scalar.
        # scalar_t = (sum_{i,j} u_{ij} A_{ij}) (sum_{j,k} v_{jk} B_{jk})
        # Then C += scalar_t * W_t (as a matrix)?
        # No, W_t is a vector of length 24.
        # So C += scalar_t * W_t.reshape(6, 4)?
        # But C is 4x6. W_t is 6x4 (k*n+i).
        # So C^T += scalar_t * W_t.
        
        scalar = np.sum(U_t * A_test) * np.sum(V_t * B_test)
        C_calc.T += scalar * W_t
        
    # Check error
    error = np.max(np.abs(C_calc - C_test))
    if error > 1e-9:
        # Fallback to schoolbook
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i * 4 + j, col] = 1
                    V[j * 6 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
    else:
        # Convert to lists of ints/floats
        U = U.tolist()
        V = V.tolist()
        W = W.tolist()
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
