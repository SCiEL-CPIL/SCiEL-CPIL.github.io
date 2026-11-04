import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank-98 decomposition.
    # It consists of two rank-49 decompositions for 4x4x4.
    # One for the first 4 columns of B, one for the last 4.
    
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a rank-49 decomposition for 4x4x4
    # offset_k: 0 for first 4 cols, 4 for last 4 cols
    # col_offset: 0 for first 49 cols, 49 for last 49 cols
    
    def add_4x4x4_decomp(offset_k, col_offset):
        # The 7 Strassen products for 2x2x2
        # Each is a rank-7 decomposition
        # We need to map the 2x2 indices to 4x4 indices
        
        # Strassen factors for 2x2x2
        # U_2 (4x7), V_2 (4x7), W_2 (4x7)
        # Indices for U_2: i*2+j
        # Indices for V_2: j*2+k
        # Indices for W_2: k*2+i
        
        U_2 = np.zeros((4, 7))
        V_2 = np.zeros((4, 7))
        W_2 = np.zeros((4, 7))
        
        # M1: (a+d)(e+h)
        U_2[0, 0] = 1; U_2[3, 0] = 1
        V_2[0, 0] = 1; V_2[3, 0] = 1
        W_2[0, 0] = 1; W_2[3, 0] = 1 # C11, C22
        
        # M2: (c+d)(e+f)
        U_2[2, 1] = 1; U_2[3, 1] = 1
        V_2[0, 1] = 1; V_2[1, 1] = 1
        W_2[0, 1] = 0; W_2[1, 1] = 0; W_2[2, 1] = -1; W_2[3, 1] = 1 # C21, C22
        
        # M3: (a+b)h
        U_2[0, 2] = 1; U_2[1, 2] = 1
        V_2[2, 2] = 1; V_2[3, 2] = 1
        W_2[0, 2] = 0; W_2[1, 2] = 1; W_2[2, 2] = 0; W_2[3, 2] = 1 # C12, C22
        
        # M4: a(f-h)
        U_2[0, 3] = 1
        V_2[1, 3] = 1; V_2[3, 3] = -1
        W_2[0, 3] = 1; W_2[1, 3] = 1; W_2[2, 3] = -1; W_2[3, 3] = 0 # C11, C12, C21
        
        # M5: d(g-e)
        U_2[3, 4] = 1
        V_2[2, 4] = 1; V_2[0, 4] = -1
        W_2[0, 4] = -1; W_2[1, 4] = 0; W_2[2, 4] = 1; W_2[3, 4] = 0 # C11, C21
        
        # M6: (b-d)(e+g)
        U_2[1, 5] = 1; U_2[3, 5] = -1
        V_2[0, 5] = 1; V_2[2, 5] = 1
        W_2[0, 5] = 0; W_2[1, 5] = -1; W_2[2, 5] = 0; W_2[3, 5] = 1 # C12, C22
        
        # M7: (c-a)(f+h)
        U_2[2, 6] = 1; U_2[0, 6] = -1
        V_2[1, 6] = 1; V_2[3, 6] = 1
        W_2[0, 6] = 1; W_2[1, 6] = 1; W_2[2, 6] = -1; W_2[3, 6] = -1 # C11, C12, C21, C22

        # Now, the 4x4x4 decomposition uses 7 such 2x2x2 decompositions
        # corresponding to the 7 Strassen products of the 2x2 blocks of A and B.
        
        # Block indices for A: 00, 01, 10, 11
        # Block indices for B: 00, 01, 10, 11
        
        # M1_4 = (A00+A11)(B00+B11)
        # M2_4 = A10(B00+B01)
        # M3_4 = (A00+A01)B11
        # M4_4 = A00(B01-B11)
        # M5_4 = A11(B10-B00)
        # M6_4 = (A01-A00)(B00+B10)
        # M7_4 = (A10-A11)(B01+B11)
        
        # We need to map the 2x2 indices to 4x4 indices.
        # For each of the 7 M_4 products, we have a 2x2x2 multiplication.
        # The "A" part of M_4 is a 2x2 matrix formed from blocks of A.
        # The "B" part of M_4 is a 2x2 matrix formed from blocks of B.
        
        # Let's define the mapping for each of the 7 products.
        # Product k (0..6)
        # A_part: 2x2 matrix of coefficients for A blocks
        # B_part: 2x2 matrix of coefficients for B blocks
        
        # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
        # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
        
        # M1_4: (A00+A11)(B00+B11)
        # A_part: [[1, 0], [0, 1]] -> A00, A11
        # B_part: [[1, 0], [0, 1]] -> B00, B11
        
        # M2_4: A10(B00+B01)
        # A_part: [[0, 0], [1, 0]] -> A10
        # B_part: [[1, 1], [0, 0]] -> B00, B01
        
        # M3_4: (A00+A01)B11
        # A_part: [[1, 1], [0, 0]] -> A00, A01
        # B_part: [[0, 0], [0, 1]] -> B11
        
        # M4_4: A00(B01-B11)
        # A_part: [[1, 0], [0, 0]] -> A00
        # B_part: [[0, 1], [0, -1]] -> B01, B11
        
        # M5_4: A11(B10-B00)
        # A_part: [[0, 0], [0, 1]] -> A11
        # B_part: [[-1, 0], [1, 0]] -> B00, B10
        
        # M6_4: (A01-A00)(B00+B10)
        # A_part: [[-1, 1], [0, 0]] -> A00, A01
        # B_part: [[1, 0], [1, 0]] -> B00, B10
        
        # M7_4: (A10-A11)(B01+B11)
        # A_part: [[0, 0], [1, -1]] -> A10, A11
        # B_part: [[0, 1], [0, 1]] -> B01, B11

        A_parts = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 0]],
            [[1, 1], [0, 0]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[-1, 1], [0, 0]],
            [[0, 0], [1, -1]]
        ]
        
        B_parts = [
            [[1, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]]
        ]
        
        for k in range(7):
            # For each k, we have a 2x2x2 multiplication
            # We use the U_2, V_2, W_2 factors
            # But we need to scale them by the coefficients in A_parts[k] and B_parts[k]
            
            # U_2 has indices i*2+j for i,j in 0..1
            # These correspond to the 2x2 blocks of A.
            # Block (bi, bj) of A corresponds to global A indices:
            # i_global = bi*2 + i_local
            # j_global = bj*2 + j_local
            
            # V_2 has indices j*2+k for j,k in 0..1
            # These correspond to the 2x2 blocks of B.
            # Block (bj, bk) of B corresponds to global B indices:
            # j_global = bj*2 + j_local
            # k_global = bk*2 + k_local + offset_k (since B is 4x8, we have two sets of 4 cols)
            
            # W_2 has indices k*2+i for k,i in 0..1
            # These correspond to the 2x2 blocks of C.
            # Block (bk, bi) of C corresponds to global C indices:
            # k_global = bk*2 + k_local + offset_k
            # i_global = bi*2 + i_local
            
            for u_idx in range(4):
                i_loc = u_idx // 2
                j_loc = u_idx % 2
                # u_idx is i_loc*2 + j_loc in the 2x2 block
                # We need to sum over the 2x2 blocks of A
                # A_parts[k] is a 2x2 matrix where entry (bi, bj) is the coefficient
                # for the block A_bj_bi? No, A is 4x4.
                # A_parts[k] defines the linear combination of the 4 blocks.
                # Let's say A_parts[k][bi][bj] is the coeff for block A_{bi, bj}.
                # Wait, the standard Strassen for 4x4 uses:
                # M1 = (A00+A11)(B00+B11).
                # A00 is top-left. A11 is bottom-right.
                # So A_parts[0] = [[1, 0], [0, 1]] means coeff 1 for A00, 0 for A01, 0 for A10, 1 for A11.
                
                coeff = 0
                for bi in range(2):
                    for bj in range(2):
                        # A_parts[k][bi][bj]
                        c = A_parts[k][bi][bj]
                        if c != 0:
                            # The local index u_idx corresponds to i_loc, j_loc within the block.
                            # The global index is bi*2 + i_loc, bj*2 + j_loc.
                            # U index is i_global * m + j_global = (bi*2+i_loc)*4 + (bj*2+j_loc)
                            u_global = (bi*2 + i_loc) * 4 + (bj*2 + j_loc)
                            coeff += c * U_2[u_idx, k]
                U[u_global, col_offset + k*7 + (u_idx % 7)] # Wait, the rank-7 decomposition has 7 columns.
                # Actually, for each k (0..6), we have 7 rank-1 terms.
                # So the column index is col_offset + k*7 + r, where r is 0..6.
                # But U_2 has 7 columns.
                # So for a fixed k, we add 7 terms.
                # The term index is r.
                # U[u_global, col_offset + k*7 + r] += coeff * U_2[u_idx, r] ?
                # No, U_2[u_idx, r] is the factor for the r-th rank-1 term of the 2x2x2 multiplication.
                # And we are constructing the r-th rank-1 term of the 4x4x4 multiplication for block k.
                
                # Let's restructure the loop.
                # Outer loop: k (0..6)
                # Inner loop: r (0..6)
                # For each (k, r), we have a rank-1 term.
                # U_part = sum_{bi,bj} A_parts[k][bi][bj] * U_2_block[i,j, r]
                # V_part = sum_{bj,bk} B_parts[k][bj][bk] * V_2_block[j,k, r]
                # W_part = sum_{bk,bi} W_2_block[k,i, r] (no coeff, just mapping)
                
                # This logic is getting complex to implement in a single pass without pre-computing.
                # I will pre-compute the U, V, W for the 4x4x4 decomposition first.
                
            # Let's pre-compute U_444, V_444, W_444 of shape (16, 49)
            U_444 = np.zeros((16, 49))
            V_444 = np.zeros((16, 49))
            W_444 = np.zeros((16, 49))
            
            for k in range(7):
                for r in range(7):
                    col = k * 7 + r
                    # U_444
                    for bi in range(2):
                        for bj in range(2):
                            c = A_parts[k][bi][bj]
                            if c != 0:
                                for i_loc in range(2):
                                    for j_loc in range(2):
                                        u_idx = i_loc * 2 + j_loc
                                        u_global = (bi * 2 + i_loc) * 4 + (bj * 2 + j_loc)
                                        U_444[u_global, col] += c * U_2[u_idx, r]
                                        
                    # V_444
                    for bj in range(2):
                        for bk in range(2):
                            c = B_parts[k][bj][bk]
                            if c != 0:
                                for j_loc in range(2):
                                    for k_loc in range(2):
                                        v_idx = j_loc * 2 + k_loc
                                        # v_idx in V_2 is j_loc*2 + k_loc
                                        v_global = (bj * 2 + j_loc) * 4 + (bk * 2 + k_loc)
                                        V_444[v_global, col] += c * V_2[v_idx, r]
                                        
                    # W_444
                    for bk in range(2):
                        for bi in range(2):
                            # W_2 indices are k_loc*2 + i_loc
                            for k_loc in range(2):
                                for i_loc in range(2):
                                    w_idx = k_loc * 2 + i_loc
                                    w_global = (bk * 2 + k_loc) * 4 + (bi * 2 + i_loc)
                                    W_444[w_global, col] += W_2[w_idx, r]
                                    
            # Now add U_444, V_444, W_444 to the global U, V, W
            # Offset for V and W due to p=8
            # V index: j*p + k. Here j is 0..3, k is 0..3 or 4..7.
            # In V_444, the index is j*4 + k.
            # We need to map k to k + offset_k.
            # So V_global[j, k+offset_k] = V_444[j*4+k, col]
            
            for j in range(4):
                for k in range(4):
                    v_444_idx = j * 4 + k
                    v_global_idx = j * 8 + (k + offset_k)
                    V[v_global_idx, col_offset:] += V_444[v_444_idx, :]
                    
            for k in range(4):
                for i in range(4):
                    w_444_idx = k * 4 + i
                    w_global_idx = (k + offset_k) * 4 + i
                    W[w_global_idx, col_offset:] += W_444[w_444_idx, :]
                    
            U[:, col_offset:] += U_444
            
    add_4x4x4_decomp(0, 0)
    add_4x4x4_decomp(4, 49)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
