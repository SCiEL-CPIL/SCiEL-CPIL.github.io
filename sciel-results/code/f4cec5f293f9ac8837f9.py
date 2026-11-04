import numpy as np

def construct():
    n, m, p = 3, 4, 6
    
    # We will build the decomposition by splitting the dimensions.
    # A is 3x4, B is 4x6, C is 3x6.
    # Split i (3) into 2+1.
    # Split j (4) into 2+2.
    # Split k (6) into 3+3.
    
    # This creates 8 blocks.
    # Blocks (i_block, j_block, k_block)
    # i_block 0: i in [0,1] (dim 2)
    # i_block 1: i in [2] (dim 1)
    # j_block 0: j in [0,1] (dim 2)
    # j_block 1: j in [2,3] (dim 2)
    # k_block 0: k in [0,2] (dim 3)
    # k_block 1: k in [3,5] (dim 3)
    
    # For each (i_b, j_b, k_b), we compute the multiplication of:
    # A_sub (dim_i x dim_j) * B_sub (dim_j x dim_k) -> C_sub (dim_i x dim_k)
    
    # We need decompositions for:
    # 2x2x3 (Rank 11: 7+4)
    # 1x2x3 (Rank 6: 4+2)
    
    # Let's create a helper to add terms to U, V, W.
    # U shape: (12, R)
    # V shape: (24, R)
    # W shape: (18, R)
    
    U = []
    V = []
    W = []
    
    # Helper to add a decomposition for a specific block
    def add_block(i_start, i_dim, j_start, j_dim, k_start, k_dim, terms):
        # terms: list of (u_vec, v_vec, w_vec) where vectors are local
        # u_vec length i_dim * j_dim
        # v_vec length j_dim * k_dim
        # w_vec length k_dim * i_dim (Note: W index is k*n + i. Here local n is i_dim? No, global n is 3.
        # The task says W[k*n+i, r]. n=3.
        # So for a block with i_start, i_dim, k_start, k_dim:
        # Local i' in [0, i_dim), local k' in [0, k_dim).
        # Global i = i_start + i'. Global k = k_start + k'.
        # W index = k_global * 3 + i_global.
        
        for u_local, v_local, w_local in terms:
            u_global = [0.0] * (3 * 4) # n*m = 12
            v_global = [0.0] * (4 * 6) # m*p = 24
            w_global = [0.0] * (6 * 3) # p*n = 18
            
            # Fill U
            for idx in range(i_dim * j_dim):
                i_l, j_l = divmod(idx, j_dim)
                u_global[(i_start + i_l) * 4 + (j_start + j_l)] = u_local[idx]
                
            # Fill V
            for idx in range(j_dim * k_dim):
                j_l, k_l = divmod(idx, k_dim)
                v_global[(j_start + j_l) * 6 + (k_start + k_l)] = v_local[idx]
                
            # Fill W
            for idx in range(k_dim * i_dim):
                k_l, i_l = divmod(idx, i_dim) # w_local is indexed by k_dim * i_dim?
                # The tensor term is u * v * w.
                # The standard decomposition gives w as a vector of length i_dim * k_dim?
                # No, the output C is i_dim x k_dim.
                # The term contributes to C[i_l, k_l].
                # So w_local should be length i_dim * k_dim?
                # But the factor W is of shape (p*n, rank).
                # The decomposition is sum_r U[...] V[...] W[...].
                # The product U[i,j] V[j,k] W[k,i] must sum to C[i,k].
                # So W is indexed by (k, i).
                # So w_local should correspond to the (k, i) indices.
                # Length k_dim * i_dim.
                
                k_global = k_start + k_l
                i_global = i_start + i_l
                w_global[k_global * 3 + i_global] = w_local[idx]
            
            U.append(u_global)
            V.append(v_global)
            W.append(w_global)
            
    # Decomposition for 2x2x2 (Strassen)
    def get_222_terms():
        # Returns list of (u, v, w) for 2x2x2
        # u length 4, v length 4, w length 4 (indexed k*2+i)
        terms = []
        # M1 = (a+d)(e+h)
        # a=0,0 d=1,1 e=0,0 h=1,1
        # u: [0,0, 0,1] (a,d) -> indices 0, 3? 
        # A indices: 00, 01, 10, 11.
        # a=0, d=1. u = [1, 0, 0, 1]
        # B indices: 00, 01, 10, 11.
        # e=0, h=1. v = [1, 0, 0, 1]
        # C indices: 00, 01, 10, 11.
        # w: contributes to all? No.
        # C11 += M1, C12 += 0, C21 += 0, C22 += M1?
        # Wait, Strassen formulas:
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: M1 = (a+d)(e+h)
        # u1 = [1, 0, 0, 1] (a, d)
        # v1 = [1, 0, 0, 1] (e, h)
        # w1 = [1, 0, 0, 1] (C11, C22)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        
        # Term 2: M2 = (c+d)e
        # u2 = [0, 0, 1, 1] (c, d)
        # v2 = [1, 0, 0, 0] (e)
        # w2 = [0, 0, 1, -1] (C21, -C22)
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        
        # Term 3: M3 = a(f-h)
        # u3 = [1, 0, 0, 0] (a)
        # v3 = [0, 1, 0, -1] (f, -h)
        # w3 = [0, 1, 0, 1] (C12, C22)
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        
        # Term 4: M4 = d(g-e)
        # u4 = [0, 0, 0, 1] (d)
        # v4 = [0, 0, 1, -1] (g, -e) -> Wait, e is 0,0. g is 1,0.
        # v4 indices: e=0, f=1, g=2, h=3.
        # g-e: v = [ -1, 0, 1, 0 ]?
        # M4 = d(g-e).
        # u4 = [0,0,0,1] (d)
        # v4 = [-1, 0, 1, 0] (g-e)
        # w4 = [1, 0, 1, 0] (C11, C21)
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        
        # Term 5: M5 = (a+b)h
        # u5 = [1, 1, 0, 0] (a, b)
        # v5 = [0, 0, 0, 1] (h)
        # w5 = [-1, 1, 0, 0] (-C11, C12)
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        
        # Term 6: M6 = (c-d)(e+f)
        # u6 = [0, 0, 1, -1] (c, -d)
        # v6 = [1, 1, 0, 0] (e, f)
        # w6 = [0, 0, 0, 1] (C22)
        terms.append(([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]))
        
        # Term 7: M7 = (b-d)(g+h)
        # u7 = [0, 1, 0, -1] (b, -d)
        # v7 = [0, 0, 1, 1] (g, h)
        # w7 = [0, 0, 0, -1] (-C22) -> Wait.
        # C11 = M1 + M4 - M5 - M7.
        # C22 = M1 - M2 + M3 + M6.
        # My w vectors:
        # w1: C11, C22
        # w2: C21, -C22
        # w3: C12, C22
        # w4: C11, C21
        # w5: -C11, C12
        # w6: C22
        # w7: ?
        # C11 needs -M7. So w7 should have -1 at C11.
        # C22 needs 0 from M7?
        # Let's check C11: M1(1) + M4(1) - M5(-1) - M7(-1) -> 1+1+1-1 = 2? No.
        # C11 = M1 + M4 - M5 - M7.
        # Coeff of M1: 1.
        # Coeff of M4: 1.
        # Coeff of M5: -1.
        # Coeff of M7: -1.
        # My w1 has 1 at C11.
        # My w4 has 1 at C11.
        # My w5 has -1 at C11.
        # My w7 should have -1 at C11.
        # So w7 = [-1, 0, 0, 0].
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 0]))
        
        return terms

    # Decomposition for 2x2x1 (Schoolbook Rank 4)
    def get_221_terms():
        # A 2x2, B 2x1, C 2x1.
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        terms = []
        # Term 1: A00 B0 -> C0
        terms.append(([1, 0, 0, 0], [1, 0], [1, 0])) # u: A00, v: B0, w: C0
        # Term 2: A01 B1 -> C0
        terms.append(([0, 1, 0, 0], [0, 1], [1, 0]))
        # Term 3: A10 B0 -> C1
        terms.append(([0, 0, 1, 0], [1, 0], [0, 1]))
        # Term 4: A11 B1 -> C1
        terms.append(([0, 0, 0, 1], [0, 1], [0, 1]))
        return terms

    # Decomposition for 1x2x2 (Schoolbook Rank 4)
    def get_122_terms():
        # A 1x2, B 2x2, C 1x2.
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        terms = []
        terms.append(([1, 0], [1, 0, 0, 0], [1, 0])) # A0, B00, C0
        terms.append(([0, 1], [0, 0, 1, 0], [1, 0])) # A1, B10, C0
        terms.append(([1, 0], [0, 1, 0, 0], [0, 1])) # A0, B01, C1
        terms.append(([0, 1], [0, 0, 0, 1], [0, 1])) # A1, B11, C1
        return terms

    # Decomposition for 1x2x1 (Schoolbook Rank 2)
    def get_121_terms():
        # A 1x2, B 2x1, C 1x1.
        # C = A0 B0 + A1 B1
        terms = []
        terms.append(([1, 0], [1, 0], [1]))
        terms.append(([0, 1], [0, 1], [1]))
        return terms

    # Combine 2x2x3 (Rank 11)
    def get_223_terms():
        # Split k (3) into 2+1.
        # Part 1: 2x2x2 (Rank 7) for k in [0,1]
        # Part 2: 2x2x1 (Rank 4) for k=2
        terms = []
        t222 = get_222_terms()
        # Map w from 2x2 (k in 0,1) to 2x3 (k in 0,1,2)
        # w_local length 4 (k*2+i). k=0,1.
        # New w length 6 (k*2+i). k=0,1,2.
        for u, v, w in t222:
            # v length 4 (j*2+k). j=0,1. k=0,1.
            # New v length 6 (j*3+k). j=0,1. k=0,1,2.
            v_new = [0.0] * 6
            for j in range(2):
                for k in range(2):
                    v_new[j*3 + k] = v[j*2 + k]
            
            w_new = [0.0] * 6
            for k in range(2):
                for i in range(2):
                    w_new[k*2 + i] = w[k*2 + i]
            # k=2 entries are 0.
            
            terms.append((u, v_new, w_new))
            
        t221 = get_221_terms()
        # u length 4. v length 2 (j*1+k). j=0,1. k=0 (local).
        # w length 2 (k*2+i). k=0 (local). i=0,1.
        for u, v, w in t221:
            v_new = [0.0] * 6
            # Local k=0 corresponds to global k=2.
            for j in range(2):
                v_new[j*3 + 2] = v[j]
                
            w_new = [0.0] * 6
            # Local k=0 corresponds to global k=2.
            for i in range(2):
                w_new[2*2 + i] = w[i]
                
            terms.append((u, v_new, w_new))
            
        return terms

    # Combine 1x2x3 (Rank 6)
    def get_123_terms():
        # Split k (3) into 2+1.
        # Part 1: 1x2x2 (Rank 4) for k in [0,1]
        # Part 2: 1x2x1 (Rank 2) for k=2
        terms = []
        t122 = get_122_terms()
        for u, v, w in t122:
            v_new = [0.0] * 6
            for j in range(2):
                for k in range(2):
                    v_new[j*3 + k] = v[j*2 + k]
            w_new = [0.0] * 3 # k*1+i. k=0,1,2. i=0.
            for k in range(2):
                w_new[k] = w[k] # w was length 2 (k*1+i)
            terms.append((u, v_new, w_new))
            
        t121 = get_121_terms()
        for u, v, w in t121:
            v_new = [0.0] * 6
            for j in range(2):
                v_new[j*3 + 2] = v[j]
            w_new = [0.0] * 3
            w_new[2] = w[0]
            terms.append((u, v_new, w_new))
            
        return terms

    # Now add all 8 blocks
    # i_blocks: (0, 2), (2, 1)
    # j_blocks: (0, 2), (2, 2)
    # k_blocks: (0, 3), (3, 3)
    
    blocks = [
        (0, 2, 0, 2, 0, 3, get_223_terms),
        (0, 2, 2, 2, 0, 3, get_223_terms),
        (0, 2, 0, 2, 3, 3, get_223_terms),
        (0, 2, 2, 2, 3, 3, get_223_terms),
        (2, 1, 0, 2, 0, 3, get_123_terms),
        (2, 1, 2, 2, 0, 3, get_123_terms),
        (2, 1, 0, 2, 3, 3, get_123_terms),
        (2, 1, 2, 2, 3, 3, get_123_terms),
    ]
    
    for i_s, i_d, j_s, j_d, k_s, k_d, getter in blocks:
        terms = getter()
        add_block(i_s, i_d, j_s, j_d, k_s, k_d, terms)
        
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
