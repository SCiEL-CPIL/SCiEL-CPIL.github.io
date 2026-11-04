import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Generates terms for N x N x N multiplication
    def get_strassen_terms(N):
        if N == 1:
            # u, v, w are vectors of size 1
            return [([1], [1], [1])]
        
        half = N // 2
        # Get terms for half size
        sub_terms = get_strassen_terms(half)
        
        # Indices for N x N matrices
        # A is N x N, flattened row-major.
        # Block A_11 is top-left half x half.
        
        # Strassen formulas for 2x2 blocks of size half x half
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map the sub-terms (which are for size half)
        # to the full size N vectors.
        
        terms = []
        
        # Helper to create full size vector from block contributions
        # block_idx: 0=A11, 1=A12, 2=A21, 3=A22
        # coeff: scalar multiplier for the block
        def make_full_vec(block_indices, coeffs, size):
            # size is N
            # block_indices is list of block indices (0-3)
            # coeffs is list of coeffs
            vec = [0] * (size * size)
            for b_idx, c in zip(block_indices, coeffs):
                # Get local indices for this block
                # Block 0: rows 0..half-1, cols 0..half-1
                # Block 1: rows 0..half-1, cols half..N-1
                # Block 2: rows half..N-1, cols 0..half-1
                # Block 3: rows half..N-1, cols half..N-1
                
                # We need to add 'c' * (sub_vector) to the appropriate part
                # But sub_vector is size half x half.
                # This function is for constructing U, V, W vectors.
                # The sub_terms provide u_sub, v_sub, w_sub of size half*half.
                # We need to place them into the N*N vector.
                pass
            
            return vec

        # This approach is slightly complex to vectorize directly in the recursive step
        # because the "addition" of blocks happens in the U and V vectors,
        # while the "addition" of results happens in the W vector.
        
        # Let's do it explicitly for N=2 (base case for N=4)
        if N == 2:
            # sub_terms are for N=1
            # u_sub, v_sub, w_sub are [1]
            terms = []
            # M1
            u = [1, 0, 0, 1] # A11, A22
            v = [1, 0, 0, 1] # B11, B22
            w = [1, 0, 0, 1] # C11, C22
            terms.append((u, v, w))
            
            # M2
            u = [0, 0, 1, 1] # A21, A22
            v = [1, 0, 0, 0] # B11
            w = [0, 0, 1, 0] # C21
            terms.append((u, v, w))
            
            # M3
            u = [1, 0, 0, 0] # A11
            v = [0, 1, 0, -1] # B12 - B22
            w = [0, 1, 0, 1] # C12, C22
            terms.append((u, v, w))
            
            # M4
            u = [0, 0, 0, 1] # A22
            v = [0, 0, 1, -1] # B21 - B11
            w = [1, 0, 1, 0] # C11, C21
            terms.append((u, v, w))
            
            # M5
            u = [1, 1, 0, 0] # A11, A12
            v = [0, 0, 0, 1] # B22
            w = [-1, 1, 0, 0] # -C11, C12
            terms.append((u, v, w))
            
            # M6
            u = [0, 0, 1, -1] # A21 - A11
            v = [1, 1, 0, 0] # B11, B12
            w = [0, 0, 0, 1] # C22
            terms.append((u, v, w))
            
            # M7
            u = [0, 1, 0, -1] # A12 - A22
            v = [0, 0, 1, 1] # B21, B22
            w = [1, 0, 0, 0] # C11
            terms.append((u, v, w))
            
            return terms

        if N == 4:
            # Use the N=2 terms as "blocks"
            # Each term in N=2 list represents a multiplication of 2x2 matrices.
            # We need to expand each of these 7 terms into 7 scalar multiplications.
            # So we iterate over the 7 N=2 terms.
            # For each N=2 term (u2, v2, w2), u2 is a 4-vector representing coefficients for the 4 blocks of A.
            # We need to create a 16-vector u4.
            # The blocks are 2x2.
            # Block 0 (A11) corresponds to u2[0].
            # Block 1 (A12) corresponds to u2[1].
            # Block 2 (A21) corresponds to u2[2].
            # Block 3 (A22) corresponds to u2[3].
            
            # We get the scalar terms for 2x2 (which are actually 1x1 in the next level? No.)
            # Wait, N=2 terms use N=1 sub-terms.
            # The N=1 terms are just [1].
            # So for N=4, we use the N=2 structure.
            # The "sub-terms" for N=4 are the 7 terms of N=2?
            # No.
            # The N=2 Strassen computes 2x2 matrix mult using 7 scalar mults.
            # The N=4 Strassen computes 4x4 matrix mult using 7 "2x2 matrix mults".
            # Each "2x2 matrix mult" is itself a rank 7 tensor.
            # So we take each of the 7 N=2 terms (which define the linear combination for a 2x2 mult)
            # and replace the "scalar multiplication" with the 7-term 2x2 decomposition.
            
            n2_terms = get_strassen_terms(2)
            n1_terms = get_strassen_terms(1) # Just for clarity, but we know they are [1]
            
            full_terms = []
            
            # Map block index to range in 16-vector
            # Block 0: 0-3
            # Block 1: 4-7
            # Block 2: 8-11
            # Block 3: 12-15
            
            for u2, v2, w2 in n2_terms:
                # u2 has 4 entries: coeff for A11, A12, A21, A22
                # For each block with non-zero coeff, we add the corresponding 2x2 term.
                # But wait, the 2x2 term (from n2_terms? No, from n1_terms?)
                # The "multiplication" in the N=2 formula is a 2x2 matrix multiplication.
                # So we need the 7 terms for 2x2 matrix multiplication.
                # Those are exactly n2_terms? No, n2_terms are for 2x2.
                # Yes, n2_terms are the 7 rank-1 tensors for 2x2x2.
                
                # So for each of the 7 "big" terms (u2, v2, w2):
                # We iterate over the 7 "small" terms (us, vs, ws) from n2_terms.
                # And combine them.
                
                for us, vs, ws in n2_terms:
                    # us, vs, ws are size 4 vectors (for 2x2 matrices).
                    # We need to create size 16 vectors u4, v4, w4.
                    
                    u4 = [0] * 16
                    v4 = [0] * 16
                    w4 = [0] * 16
                    
                    # u2[0] is coeff for A11.
                    # us[0] is coeff for A11 (within the 2x2 block).
                    # So u4[0..3] += u2[0] * us
                    # u4[4..7] += u2[1] * us
                    # etc.
                    
                    for b in range(4):
                        if u2[b] != 0:
                            for i in range(4):
                                u4[b*4 + i] += u2[b] * us[i]
                        if v2[b] != 0:
                            for i in range(4):
                                v4[b*4 + i] += v2[b] * vs[i]
                        if w2[b] != 0:
                            for i in range(4):
                                w4[b*4 + i] += w2[b] * ws[i]
                    
                    # Check if all zero
                    if any(x != 0 for x in u4) and any(x != 0 for x in v4) and any(x != 0 for x in w4):
                        full_terms.append((u4, v4, w4))
            
            return full_terms

    terms_444 = get_strassen_terms(4)
    rank_444 = len(terms_444) # Should be 49
    
    # Part 2: 4x4x3 Schoolbook (Rank 48)
    terms_443 = []
    for i in range(4):
        for j in range(4):
            for k in range(3): # k is 0,1,2 local to the 3 cols
                u = [0] * 16
                v = [0] * 16 # Note: V for 4x4x3 part is size 4x3=12? No, V is size m*p.
                # In the combined tensor, V is size m*p = 4*7 = 28.
                # We will handle global indexing later.
                # Here we just generate local terms.
                
                # Local indices for 4x4x3:
                # U: 16 (4x4)
                # V: 12 (4x3)
                # W: 12 (3x4)
                
                u[4*i + j] = 1
                v_local_idx = j * 3 + k
                # w_local_idx = k * 4 + i
                
                # We store local indices and coeff
                terms_443.append((4*i + j, v_local_idx, k*4 + i, 1))
    
    rank_443 = len(terms_443) # 48
    
    # Construct Global U, V, W
    total_rank = rank_444 + rank_443
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Add 4x4x4 terms
    for u4, v4, w4 in terms_444:
        # u4, v4, w4 are lists of 16 floats/ints
        for idx in range(16):
            if u4[idx] != 0:
                i, j = divmod(idx, 4)
                U[i * 4 + j][col] += u4[idx]
            if v4[idx] != 0:
                j, k = divmod(idx, 4)
                V[j * 7 + k][col] += v4[idx]
            if w4[idx] != 0:
                k, i = divmod(idx, 4)
                W[k * 4 + i][col] += w4[idx]
        col += 1
        
    # Add 4x4x3 terms
    for u_idx, v_idx, w_idx, coeff in terms_443:
        # u_idx is 0..15
        i, j = divmod(u_idx, 4)
        U[i * 4 + j][col] += coeff
        
        # v_idx is 0..11 (local to 4x3)
        # Global V index: j * 7 + k_global
        # v_idx = j * 3 + k_local
        j, k_local = divmod(v_idx, 3)
        k_global = k_local + 4 # Offset by 4
        V[j * 7 + k_global][col] += coeff
        
        # w_idx is 0..11 (local to 3x4)
        # Global W index: k_global * 4 + i
        k_local, i = divmod(w_idx, 4)
        k_global = k_local + 4
        W[k_global * 4 + i][col] += coeff
        
        col += 1
        
    return {"U": U, "V": V, "W": W}
