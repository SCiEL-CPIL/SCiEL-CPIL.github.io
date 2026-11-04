import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition as a list of (u, v, w) triples
    # u is of length n*m (16)
    # v is of length m*p (36)
    # w is of length p*n (36)
    
    decompositions = []
    
    # Helper to add a term
    def add_term(u, v, w):
        decompositions.append((u, v, w))
        
    # 1. Decompose A B1 (4x4 * 4x4) using Strassen
    # B1 corresponds to columns 0-3 of B
    # Indices for B1 in V: j*p + k where k in 0..3
    # Indices for W1 in W: k*n + i where k in 0..3
    
    def strassen_2x2_decomp(base_u, base_v, base_w):
        """
        Generates rank-7 decomposition for 2x2x2.
        base_u: offset for A indices (2x2 block)
        base_v: offset for B indices (2x2 block)
        base_w: offset for C indices (2x2 block)
        Returns list of (u, v, w)
        """
        terms = []
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = [0]*4; u1[0]=1; u1[3]=1
        v1 = [0]*4; v1[0]=1; v1[3]=1
        w1 = [0]*4; w1[0]=1; w1[3]=1; w1[1]=-1; w1[2]=-1 # C00 += M1, C11 += M1
        # Wait, w vector is for C.
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So w for M1 should have 1 at C00 and 1 at C11?
        # No, the term is u * v * w.
        # If w has 1 at C00 and 1 at C11, then it adds to both.
        # But in Strassen, C00 gets +M1, C11 gets +M1.
        # So w1 = [1, 0, 0, 1] for C00, C01, C10, C11?
        # C indices: 0->C00, 1->C01, 2->C10, 3->C11.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C00 and +1 to C11.
        
        def make_vec(indices_values):
            vec = [0]*4
            for idx, val in indices_values:
                vec[idx] += val
            return vec
        
        # M1
        u1 = make_vec([(0,1), (3,1)])
        v1 = make_vec([(0,1), (3,1)])
        w1 = make_vec([(0,1), (3,1)])
        terms.append((u1, v1, w1))
        
        # M2 = (A10 + A11) B00
        u2 = make_vec([(2,1), (3,1)])
        v2 = make_vec([(0,1)])
        w2 = make_vec([(2,1), (3,-1)]) # C10 += M2, C11 -= M2
        terms.append((u2, v2, w2))
        
        # M3 = A00 (B10 - B11)
        u3 = make_vec([(0,1)])
        v3 = make_vec([(2,1), (3,-1)])
        w3 = make_vec([(1,1), (3,1)]) # C01 += M3, C11 += M3
        terms.append((u3, v3, w3))
        
        # M4 = A11 (B01 - B00)
        u4 = make_vec([(3,1)])
        v4 = make_vec([(1,1), (0,-1)])
        w4 = make_vec([(0,1), (2,1)]) # C00 += M4, C10 += M4
        terms.append((u4, v4, w4))
        
        # M5 = (A00 + A01) B11
        u5 = make_vec([(0,1), (1,1)])
        v5 = make_vec([(3,1)])
        w5 = make_vec([(0,-1), (1,1)]) # C00 -= M5, C01 += M5
        terms.append((u5, v5, w5))
        
        # M6 = (A10 - A00) (B00 + B01)
        u6 = make_vec([(2,1), (0,-1)])
        v6 = make_vec([(0,1), (1,1)])
        w6 = make_vec([(3,1)]) # C11 += M6
        terms.append((u6, v6, w6))
        
        # M7 = (A01 - A11) (B10 + B11)
        u7 = make_vec([(1,1), (3,-1)])
        v7 = make_vec([(2,1), (3,1)])
        w7 = make_vec([(0,1)]) # C00 += M7
        terms.append((u7, v7, w7))
        
        return terms

    # We need to map local 2x2 indices to global 4x4 indices.
    # A is 4x4. Blocks:
    # A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B is 4x4 (for B1). Blocks:
    # B00: rows 0-1, cols 0-1 -> indices 0,1,4,5 (in B1 local)
    # But in global V (4x9), B1 is cols 0-3.
    # V indices: j*9 + k.
    # B1 local (j,k) -> global (j, k).
    
    # C is 4x4 (for C1). Blocks:
    # C00: rows 0-1, cols 0-1 -> indices 0,1,4,5 (in C1 local)
    # W indices: k*4 + i.
    # C1 local (k,i) -> global (k, i).
    
    # We will generate the 49 terms for A B1.
    # Strassen 4x4 uses 7 Strassen 2x2.
    # Each Strassen 2x2 uses 7 scalar mults.
    
    # Let's just generate the 49 terms directly by nesting.
    
    # This is getting complex to hardcode the mapping.
    # Alternative: Use the 114 rank decomposition but generate it via a small script logic in Python.
    
    # Given the time, I will implement the 114-rank decomposition.
    # It is safer than trying to hardcode Strassen indices incorrectly.
    
    # Reset decompositions
    decompositions = []
    
    # Part 1: A B1 (4x4 * 4x4) using Strassen
    # We need the rank-49 decomposition of M_4,4,4.
    # I will construct it using the rank-7 decomposition of M_2,2,2.
    
    def get_strassen_4x4_terms():
        """
        Returns list of (u, v, w) for 4x4x4 Strassen.
        u: len 16, v: len 16, w: len 16.
        """
        terms = []
        # We have 7 top-level terms.
        # Each top-level term is a product of two 2x2 matrices.
        # We use strassen_2x2_decomp for each.
        
        # Top level mappings:
        # T1: (A00+A11) * (B00+B11)
        # T2: (A10+A11) * B00
        # T3: A00 * (B10-B11)
        # T4: A11 * (B01-B00)
        # T5: (A00+A01) * B11
        # T6: (A10-A00) * (B00+B01)
        # T7: (A01-A11) * (B10+B11)
        
        # For each Ti, we have a linear combo of A blocks and B blocks.
        # We need to expand (sum ai * Ai_block) * (sum bj * Bj_block)
        # = sum_ij ai bj * (Ai_block * Bj_block)
        # Each Ai_block * Bj_block is a 2x2 * 2x2 multiplication.
        # We use strassen_2x2_decomp for that.
        
        # Block indices in 4x4 matrix (row-major 4x4 = 16 elements)
        # A blocks:
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        
        # B blocks (same layout)
        
        # C blocks (same layout)
        
        # Helper to create a block vector
        def block_vec(block_id, val):
            vec = [0]*16
            if block_id == 0: # 00
                vec[0]=val; vec[1]=val; vec[4]=val; vec[5]=val
            elif block_id == 1: # 01
                vec[2]=val; vec[3]=val; vec[6]=val; vec[7]=val
            elif block_id == 2: # 10
                vec[8]=val; vec[9]=val; vec[12]=val; vec[13]=val
            elif block_id == 3: # 11
                vec[10]=val; vec[11]=val; vec[14]=val; vec[15]=val
            return vec
        
        # Top level coefficients
        top_terms = [
            # (A_coeffs, B_coeffs)
            ([(0,1), (3,1)], [(0,1), (3,1)]), # T1
            ([(2,1), (3,1)], [(0,1)]),         # T2
            ([(0,1)], [(2,1), (3,-1)]),       # T3
            ([(3,1)], [(1,1), (0,-1)]),       # T4
            ([(0,1), (1,1)], [(3,1)]),        # T5
            ([(2,1), (0,-1)], [(0,1), (1,1)]),# T6
            ([(1,1), (3,-1)], [(2,1), (3,1)]) # T7
        ]
        
        for a_coeffs, b_coeffs in top_terms:
            for (ba, va) in a_coeffs:
                for (bb, vb) in b_coeffs:
                    # Multiply block ba by block bb
                    # Use strassen_2x2_decomp
                    # We need to map the 2x2 local indices to global 4x4 indices.
                    # The 2x2 decomp returns u,v,w of length 4.
                    # We need to expand these to length 16.
                    
                    sub_terms = strassen_2x2_decomp()
                    for su, sv, sw in sub_terms:
                        gu = [0]*16
                        gv = [0]*16
                        gw = [0]*16
                        
                        # Map local 0->0, 1->1, 2->2, 3->3 for the block
                        # Block ba starts at specific indices.
                        # Block 0 (00): 0,1,4,5. Local 0->0, 1->1, 2->4, 3->5
                        # Block 1 (01): 2,3,6,7. Local 0->2, 1->3, 2->6, 3->7
                        # Block 2 (10): 8,9,12,13. Local 0->8, 1->9, 2->12, 3->13
                        # Block 3 (11): 10,11,14,15. Local 0->10, 1->11, 2->14, 3->15
                        
                        def map_block(local_vec, block_id, scale):
                            res = [0]*16
                            if block_id == 0:
                                res[0] = local_vec[0]*scale
                                res[1] = local_vec[1]*scale
                                res[4] = local_vec[2]*scale
                                res[5] = local_vec[3]*scale
                            elif block_id == 1:
                                res[2] = local_vec[0]*scale
                                res[3] = local_vec[1]*scale
                                res[6] = local_vec[2]*scale
                                res[7] = local_vec[3]*scale
                            elif block_id == 2:
                                res[8] = local_vec[0]*scale
                                res[9] = local_vec[1]*scale
                                res[12] = local_vec[2]*scale
                                res[13] = local_vec[3]*scale
                            elif block_id == 3:
                                res[10] = local_vec[0]*scale
                                res[11] = local_vec[1]*scale
                                res[14] = local_vec[2]*scale
                                res[15] = local_vec[3]*scale
                            return res
                        
                        gu = map_block(su, ba, va)
                        gv = map_block(sv, bb, vb)
                        gw = map_block(sw, ba, va) # Wait, C block is determined by A and B blocks?
                        # No, the C block for A_ba * B_bb is not simply ba or bb.
                        # In Strassen, the C blocks are linear combinations.
                        # The strassen_2x2_decomp returns w for the C 2x2 block.
                        # Which C block does this contribute to?
                        # The top-level term T_k contributes to specific C blocks.
                        # T1 contributes to C00 and C11.
                        # T2 contributes to C10 and C11.
                        # T3 contributes to C01 and C11.
                        # T4 contributes to C00 and C10.
                        # T5 contributes to C00 and C01.
                        # T6 contributes to C11.
                        # T7 contributes to C00.
                        
                        # So we need to know which C block this sub-term belongs to.
                        # This mapping is complex.
                        pass
                        
        return terms

    # This is too complicated to get right quickly.
    # I will stick to the 114-rank decomposition using a simpler block method if possible, or just schoolbook 144 if I can't guarantee correctness of Strassen.
    # But 144 is the baseline. I need to beat it.
    # The 132-rank (3x4x11) is safer.
    # Let's implement the 132-rank decomposition.
    
    decompositions = []
    
    # 3 blocks of B: B0 (4x3), B1 (4x3), B2 (4x3)
    # For each block, use 4 sub-blocks of 2x2 * 2x3.
    # Each 2x2 * 2x3 uses rank 11 (4 for 2x2x1 + 7 for 2x2x2).
    
    # We need rank-11 decomp for 2x2x3.
    # M_2,2,3 = M_2,2,1 + M_2,2,2.
    # M_2,2,1: 4 terms.
    # M_2,2,2: 7 terms (Strassen).
    
    def get_2x2x3_terms():
        terms = []
        # M_2,2,1: C_k = sum_j A_jk B_j0? No.
        # C is 2x3. C_i0 is col 0.
        # C_i0 = A_i0 B_00 + A_i1 B_10.
        # 4 terms: (A_i0, B_00, C_i0) and (A_i1, B_10, C_i0).
        # i in 0,1.
        
        # Indices for 2x2 A: 0,1,2,3
        # Indices for 2x3 B: 0,1,2 (row 0), 3,4,5 (row 1)
        # Indices for 2x3 C: 0,1,2 (row 0), 3,4,5 (row 1)
        
        # M_2,2,1 (col 0)
        for i in range(2):
            for j in range(2):
                u = [0]*4; u[j*2+i] = 1 # A_i,j? No, A is 2x2. A[i,j]. Index i*2+j?
                # Let's use row-major: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3.
                u = [0]*4; u[i*2+j] = 1
                v = [0]*6; v[j*3+0] = 1 # B_j,0
                w = [0]*6; w[i*3+0] = 1 # C_i,0
                terms.append((u, v, w))
                
        # M_2,2,2 (cols 1,2)
        # Use Strassen for 2x2x2 on cols 1,2.
        # B' is 2x2 (cols 1,2). C' is 2x2 (cols 1,2).
        # Map B' index (j,k') to B index (j, k'+1).
        # Map C' index (i,k') to C index (i, k'+1).
        
        strassen_terms = strassen_2x2_decomp()
        for su, sv, sw in strassen_terms:
            u = [0]*4
            v = [0]*6
            w = [0]*6
            for l in range(4):
                u[l] = su[l]
                # sv[l] is B'_j,k'. j=l//2, k'=l%2.
                j = l // 2
                kp = l % 2
                v[j*3 + (kp+1)] = sv[l]
                
                # sw[l] is C'_i,k'. i=l//2, k'=l%2.
                i = l // 2
                w[i*3 + (kp+1)] = sw[l]
            terms.append((u, v, w))
            
        return terms

    # We need to map 2x2x3 terms to global 4x4x9.
    # A is 4x4. B is 4x9. C is 9x4? No, C is 4x9. W is p*n = 9*4 = 36.
    # W[k*n + i]. k is col of C (0..8), i is row of C (0..3).
    
    # Blocks:
    # B blocks: 3 blocks of 4x3.
    # Block b (0,1,2) corresponds to cols 3b, 3b+1, 3b+2 of B.
    # A is split into 4 blocks of 2x2: A00, A01, A10, A11.
    # For each A block (ai) and B block (b), we compute C_sub = A_ai * B_b.
    # C_sub is 2x3.
    # The result goes to C rows (2*ai_row, 2*ai_row+1) and cols (3b, 3b+1, 3b+2).
    
    terms_2x2x3 = get_2x2x3_terms()
    
    for b in range(3):
        for ai in range(4):
            # ai: 0->A00, 1->A01, 2->A10, 3->A11
            # A block rows: ai//2 * 2, ai//2 * 2 + 1
            # A block cols: ai%2 * 2, ai%2 * 2 + 1
            
            a_row_start = (ai // 2) * 2
            a_col_start = (ai % 2) * 2
            
            # B block cols: 3b, 3b+1, 3b+2
            b_col_start = 3*b
            
            # C block rows: same as A rows
            # C block cols: same as B cols
            
            for su, sv, sw in terms_2x2x3:
                gu = [0]*16
                gv = [0]*36
                gw = [0]*36
                
                # Map su (len 4) to gu (len 16)
                for l in range(4):
                    i_local = l // 2
                    j_local = l % 2
                    i_global = a_row_start + i_local
                    j_global = a_col_start + j_local
                    gu[i_global*4 + j_global] = su[l]
                    
                # Map sv (len 6) to gv (len 36)
                # sv index: j_local*3 + k_local
                for l in range(6):
                    j_local = l // 3
                    k_local = l % 3
                    j_global = j_local # B rows are 0..3, no offset
                    k_global = b_col_start + k_local
                    gv[j_global*9 + k_global] = sv[l]
                    
                # Map sw (len 6) to gw (len 36)
                # sw index: i_local*3 + k_local
                for l in range(6):
                    i_local = l // 3
                    k_local = l % 3
                    i_global = a_row_start + i_local
                    k_global = b_col_start + k_local
                    gw[k_global*4 + i_global] = sw[l]
                    
                decompositions.append((gu, gv, gw))
                
    # Build U, V, W
    rank = len(decompositions)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for c, (u, v, w) in enumerate(decompositions):
        for i in range(16):
            U[i][c] = u[i]
        for j in range(36):
            V[j][c] = v[j]
        for k in range(36):
            W[k][c] = w[k]
            
    return {"U": U, "V": V, "W": W}

