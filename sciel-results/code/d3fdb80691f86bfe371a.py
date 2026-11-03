import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # We will build the decomposition for 4x4x9
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A B1 + A B2 + A B3
    
    # U, V, W matrices
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (36, rank)
    # W shape: (p*n, rank) = (36, rank)
    
    # We'll collect terms as lists of (u, v, w)
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # 1. Decomposition for A B1 (4x4 * 4x4)
    # Use Block Strassen (49 terms)
    # B1 is columns 0-3 of B
    
    # 2. Decomposition for A B2 (4x4 * 4x4)
    # Use Block Strassen (49 terms)
    # B2 is columns 4-7 of B
    
    # 3. Decomposition for A B3 (4x4 * 4x1)
    # Use Trivial (16 terms)
    # B3 is column 8 of B
    
    # Implementing Block Strassen for 4x4x4
    # This returns a list of (u, v, w) for the 4x4x4 case
    # u is length 16, v is length 16, w is length 16
    
    def block_strassen_4x4x4():
        bs_terms = []
        
        # 7-rank decomposition for 2x2x2
        # Indices for 2x2: 0,1,2,3
        # u, v, w are length 4
        # M1
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        # w contributions
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map 2x2 indices to 4x4 block indices
        # Block (bi, bj) in 2x2 corresponds to 4x4 block
        # A block (bi, bj) is at rows bi*2..bi*2+1, cols bj*2..bj*2+1
        # In the 16-dim vector for A, index is (bi*2+r) * 4 + (bj*2+c)
        # Wait, A is 4x4, flattened row-major?
        # U is (n*m, rank) -> (16, rank).
        # Index in U: i*m + j = i*4 + j.
        
        # Let's define a function to expand a 2x2 vector to 4x4 block vector
        def expand_2x2_to_4x4(vec, block_i, block_j):
            # vec is length 4: [a, b, c, d] for [[a,b],[c,d]]
            # block_i, block_j in {0,1}
            res = [0] * 16
            for r in range(2):
                for c in range(2):
                    val = vec[r*2 + c]
                    if val != 0:
                        # Row in 4x4: block_i*2 + r
                        # Col in 4x4: block_j*2 + c
                        idx = (block_i*2 + r) * 4 + (block_j*2 + c)
                        res[idx] = val
            return res
            
        # Strassen terms for 2x2
        # M1 = (A11 + A22)(B11 + B22)
        # A11 is block (0,0), A22 is block (1,1)
        # B11 is block (0,0), B22 is block (1,1)
        
        # u for M1: A11 + A22
        # v for M1: B11 + B22
        # w for M1: contributes to C11, C22?
        # Actually, the 7-rank decomposition gives 7 terms.
        # Each term is a rank-1 tensor for 2x2x2.
        # We need to "lift" each of these 7 terms to 4x4x4.
        
        # The 7 terms of 2x2x2:
        # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (This is not quite right, w depends on C)
        # Let's use the explicit formulas.
        
        # M1: u_A = A11+A22, u_B = B11+B22
        # M2: u_A = A21+A22, u_B = B11
        # M3: u_A = A11, u_B = B12-B22
        # M4: u_A = A22, u_B = B21-B11
        # M5: u_A = A11+A12, u_B = B22
        # M6: u_A = A21-A11, u_B = B11+B12
        # M7: u_A = A12-A22, u_B = B21+B22
        
        # For each Mk, we need to expand u_A and u_B to 4x4 vectors.
        # And we need to determine the w vector for 4x4.
        # The w vector for 4x4 comes from the w vectors of the 2x2 decomposition.
        # The 2x2 decomposition is:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for each Mk, we have a contribution to C11, C12, C21, C22.
        # Each Cij is a 2x2 block.
        # So for Mk, the w vector in 4x4 is the concatenation of the w vectors for the 2x2 blocks.
        
        # Let's define the 7 terms of 2x2x2 as (u, v, w)
        # u, v, w are length 4.
        
        # Term 1 (M1):
        # u: A11+A22 -> [1,0,0,1]
        # v: B11+B22 -> [1,0,0,1]
        # w: C11 gets +1, C22 gets +1. C12, C21 get 0.
        # w: [1, 0, 0, 1] (for C11, C12, C21, C22? No, w is for the 2x2 output)
        # w[0]=1 (C11), w[1]=0 (C12), w[2]=0 (C21), w[3]=1 (C22)
        
        # Term 2 (M2):
        # u: A21+A22 -> [0,0,1,1]
        # v: B11 -> [1,0,0,0]
        # w: C21 gets +1.
        # w: [0, 0, 1, 0]
        
        # Term 3 (M3):
        # u: A11 -> [1,0,0,0]
        # v: B12-B22 -> [0,1,0,-1]
        # w: C12 gets +1.
        # w: [0, 1, 0, 0]
        
        # Term 4 (M4):
        # u: A22 -> [0,0,0,1]
        # v: B21-B11 -> [0,0,1,0] - [1,0,0,0] = [-1, 0, 1, 0]
        # w: C11 gets +1, C21 gets +1.
        # w: [1, 0, 1, 0]
        
        # Term 5 (M5):
        # u: A11+A12 -> [1,1,0,0]
        # v: B22 -> [0,0,0,1]
        # w: C11 gets -1, C12 gets +1.
        # w: [-1, 1, 0, 0]
        
        # Term 6 (M6):
        # u: A21-A11 -> [0,0,1,0] - [1,0,0,0] = [-1, 0, 1, 0]
        # v: B11+B12 -> [1,1,0,0]
        # w: C22 gets +1.
        # w: [0, 0, 0, 1]
        
        # Term 7 (M7):
        # u: A12-A22 -> [0,1,0,0] - [0,0,0,1] = [0, 1, 0, -1]
        # v: B21+B22 -> [0,0,1,1]
        # w: C11 gets +1.
        # w: [1, 0, 0, 0]
        
        # Now, for each of these 7 terms, we expand u, v, w to 4x4.
        # u is for A (4x4).
        # v is for B (4x4).
        # w is for C (4x4).
        
        # Expansion:
        # u_4x4 = expand(u_2x2, block_i, block_j)
        # But u_2x2 is a sum of blocks?
        # No, u_2x2 is a vector of length 4, representing a 2x2 matrix.
        # In the 4x4 context, this 2x2 matrix is placed in a specific block?
        # No, the 2x2 matrix IS the block.
        # Wait, in Strassen, A11 is a 2x2 block.
        # So u_2x2 corresponds to the values in the block.
        # So if u_2x2 = [1, 0, 0, 1], it means A11 has 1s on diagonal.
        # So in the 4x4 vector, the entries corresponding to block (0,0) are 1,0,0,1.
        # The entries for other blocks are 0.
        
        # So for Term 1:
        # u_2x2 = [1,0,0,1] corresponds to A11+A22.
        # This means u_4x4 has [1,0,0,1] in block (0,0) and [1,0,0,1] in block (1,1).
        # So we need to sum the expansions.
        
        # Let's define a helper to add a 2x2 vector to a 4x4 vector at a specific block.
        
        def add_2x2_to_4x4(vec_4x4, vec_2x2, block_i, block_j):
            for r in range(2):
                for c in range(2):
                    val = vec_2x2[r*2 + c]
                    if val != 0:
                        idx = (block_i*2 + r) * 4 + (block_j*2 + c)
                        vec_4x4[idx] += val
                        return vec_4x4
                        # Actually, we should just loop and add.
                        
        # Let's rewrite add_2x2_to_4x4 properly.
        
        def expand_2x2_to_4x4_sum(vec_2x2_list):
            # vec_2x2_list is a list of (vec_2x2, block_i, block_j)
            res = [0] * 16
            for vec_2x2, bi, bj in vec_2x2_list:
                for r in range(2):
                    for c in range(2):
                        val = vec_2x2[r*2 + c]
                        if val != 0:
                            idx = (bi*2 + r) * 4 + (bj*2 + c)
                            res[idx] += val
            return res
            
        # Now construct the 49 terms.
        
        # Term 1 (M1)
        u1_2x2 = [1,0,0,1]
        v1_2x2 = [1,0,0,1]
        w1_2x2 = [1,0,0,1]
        
        # u1_4x4: A11 (0,0) + A22 (1,1)
        u1_4x4 = expand_2x2_to_4x4_sum([(u1_2x2, 0, 0), (u1_2x2, 1, 1)])
        # v1_4x4: B11 (0,0) + B22 (1,1)
        v1_4x4 = expand_2x2_to_4x4_sum([(v1_2x2, 0, 0), (v1_2x2, 1, 1)])
        # w1_4x4: C11 (0,0) + C22 (1,1)
        w1_4x4 = expand_2x2_to_4x4_sum([(w1_2x2, 0, 0), (w1_2x2, 1, 1)])
        
        bs_terms.append((u1_4x4, v1_4x4, w1_4x4))
        
        # Term 2 (M2)
        u2_2x2 = [0,0,1,1]
        v2_2x2 = [1,0,0,0]
        w2_2x2 = [0,0,1,0]
        
        u2_4x4 = expand_2x2_to_4x4_sum([(u2_2x2, 1, 0), (u2_2x2, 1, 1)]) # A21 (1,0) + A22 (1,1)
        v2_4x4 = expand_2x2_to_4x4_sum([(v2_2x2, 0, 0)]) # B11 (0,0)
        w2_4x4 = expand_2x2_to_4x4_sum([(w2_2x2, 1, 0)]) # C21 (1,0)
        
        bs_terms.append((u2_4x4, v2_4x4, w2_4x4))
        
        # Term 3 (M3)
        u3_2x2 = [1,0,0,0]
        v3_2x2 = [0,1,0,-1]
        w3_2x2 = [0,1,0,0]
        
        u3_4x4 = expand_2x2_to_4x4_sum([(u3_2x2, 0, 0)]) # A11 (0,0)
        v3_4x4 = expand_2x2_to_4x4_sum([(v3_2x2, 0, 1), (v3_2x2, 1, 1)]) # B12 (0,1) - B22 (1,1)
        # Wait, v3_2x2 is B12-B22.
        # B12 is block (0,1). B22 is block (1,1).
        # So v3_4x4 = expand(v3_2x2, 0, 1) + expand(-v3_2x2, 1, 1)?
        # No, v3_2x2 is the vector [0,1,0,-1].
        # This vector is added to B12 and subtracted from B22?
        # No, the term is A11 * (B12 - B22).
        # So v is B12 - B22.
        # v_4x4 has v3_2x2 in block (0,1) and -v3_2x2 in block (1,1).
        
        v3_4x4 = expand_2x2_to_4x4_sum([(v3_2x2, 0, 1), ([-x for x in v3_2x2], 1, 1)])
        w3_4x4 = expand_2x2_to_4x4_sum([(w3_2x2, 0, 1)]) # C12 (0,1)
        
        bs_terms.append((u3_4x4, v3_4x4, w3_4x4))
        
        # Term 4 (M4)
        u4_2x2 = [0,0,0,1]
        v4_2x2 = [-1,0,1,0]
        w4_2x2 = [1,0,1,0]
        
        u4_4x4 = expand_2x2_to_4x4_sum([(u4_2x2, 1, 1)]) # A22 (1,1)
        v4_4x4 = expand_2x2_to_4x4_sum([(v4_2x2, 1, 0), ([-x for x in v4_2x2], 0, 0)]) # B21 (1,0) - B11 (0,0)
        w4_4x4 = expand_2x2_to_4x4_sum([(w4_2x2, 0, 0), (w4_2x2, 1, 0)]) # C11 (0,0) + C21 (1,0)
        
        bs_terms.append((u4_4x4, v4_4x4, w4_4x4))
        
        # Term 5 (M5)
        u5_2x2 = [1,1,0,0]
        v5_2x2 = [0,0,0,1]
        w5_2x2 = [-1,1,0,0]
        
        u5_4x4 = expand_2x2_to_4x4_sum([(u5_2x2, 0, 0), (u5_2x2, 0, 1)]) # A11 (0,0) + A12 (0,1)
        v5_4x4 = expand_2x2_to_4x4_sum([(v5_2x2, 1, 1)]) # B22 (1,1)
        w5_4x4 = expand_2x2_to_4x4_sum([(w5_2x2, 0, 0), (w5_2x2, 0, 1)]) # C11 (0,0) + C12 (0,1)
        
        bs_terms.append((u5_4x4, v5_4x4, w5_4x4))
        
        # Term 6 (M6)
        u6_2x2 = [-1,0,1,0]
        v6_2x2 = [1,1,0,0]
        w6_2x2 = [0,0,0,1]
        
        u6_4x4 = expand_2x2_to_4x4_sum([(u6_2x2, 1, 0), ([-x for x in u6_2x2], 0, 0)]) # A21 (1,0) - A11 (0,0)
        v6_4x4 = expand_2x2_to_4x4_sum([(v6_2x2, 0, 0), (v6_2x2, 0, 1)]) # B11 (0,0) + B12 (0,1)
        w6_4x4 = expand_2x2_to_4x4_sum([(w6_2x2, 1, 1)]) # C22 (1,1)
        
        bs_terms.append((u6_4x4, v6_4x4, w6_4x4))
        
        # Term 7 (M7)
        u7_2x2 = [0,1,0,-1]
        v7_2x2 = [0,0,1,1]
        w7_2x2 = [1,0,0,0]
        
        u7_4x4 = expand_2x2_to_4x4_sum([(u7_2x2, 0, 1), ([-x for x in u7_2x2], 1, 1)]) # A12 (0,1) - A22 (1,1)
        v7_4x4 = expand_2x2_to_4x4_sum([(v7_2x2, 1, 0), (v7_2x2, 1, 1)]) # B21 (1,0) + B22 (1,1)
        w7_4x4 = expand_2x2_to_4x4_sum([(w7_2x2, 0, 0)]) # C11 (0,0)
        
        bs_terms.append((u7_4x4, v7_4x4, w7_4x4))
        
        return bs_terms
        
    bs_terms_1 = block_strassen_4x4x4()
    bs_terms_2 = block_strassen_4x4x4()
    
    # Now map these to the 4x4x9 problem.
    # B1 is cols 0-3. B2 is cols 4-7.
    # For B1, v indices are 0-15.
    # For B2, v indices are 16-31.
    # For B3, v indices are 32-35.
    
    # W indices:
    # C1 (4x4) is cols 0-3 of C. W indices 0-15.
    # C2 (4x4) is cols 4-7 of C. W indices 16-31.
    # C3 (4x1) is col 8 of C. W indices 32-35.
    
    # U is same for all.
    
    for u, v, w in bs_terms_1:
        # v is for B1 (cols 0-3)
        # w is for C1 (cols 0-3)
        add_term(u, v, w)
        
    for u, v, w in bs_terms_2:
        # v is for B2 (cols 4-7)
        # Shift v indices by 16
        v_shifted = [0] * 36
        for i in range(16):
            v_shifted[i+16] = v[i]
            
        # w is for C2 (cols 4-7)
        # Shift w indices by 16
        w_shifted = [0] * 36
        for i in range(16):
            w_shifted[i+16] = w[i]
            
        add_term(u, v_shifted, w_shifted)
        
    # Trivial for A B3 (4x4 * 4x1)
    # B3 is col 8.
    # C3 is col 8.
    # A is 4x4.
    # C[i, 8] = sum_j A[i, j] B[j, 8]
    # Terms: for i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            u = [0] * 16
            u[i*4 + j] = 1
            
            v = [0] * 36
            v[j*9 + 8] = 1 # B index: j*9 + k. k=8.
            
            w = [0] * 36
            w[8*4 + i] = 1 # W index: k*n + i. k=8, n=4.
            
            add_term(u, v, w)
            
    # Build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for t, (u, v, w) in enumerate(terms):
        for i in range(16):
            if u[i] != 0:
                U[i][t] = u[i]
        for i in range(36):
            if v[i] != 0:
                V[i][t] = v[i]
        for i in range(36):
            if w[i] != 0:
                W[i][t] = w[i]
                
    return {"U": U, "V": V, "W": W}
