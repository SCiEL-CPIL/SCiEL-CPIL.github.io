import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6  # A is n×m, B is m×p, C is n×p
    
    # We'll build U, V, W as lists of lists
    # U: shape (n*m, rank) = (20, rank)
    # V: shape (m*p, rank) = (30, rank)
    # W: shape (p*n, rank) = (24, rank)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u, v, w):
        """Add one rank-1 term. u is index into U, v into V, w into W."""
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)
    
    # Strassen decomposition for 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(u_base, v_base, w_base):
        """
        u_base: base index for U (2x2 block, indices u_base to u_base+3)
                 u_base+0 = (i0,j0), u_base+1 = (i0,j1), u_base+2 = (i1,j0), u_base+3 = (i1,j1)
        v_base: base index for V (2x2 block)
                 v_base+0 = (j0,k0), v_base+1 = (j0,k1), v_base+2 = (j1,k0), v_base+3 = (j1,k1)
        w_base: base index for W (2x2 block)
                 w_base+0 = (k0,i0), w_base+1 = (k0,i1), w_base+2 = (k1,i0), w_base+3 = (k1,i1)
        
        For Strassen, we need:
        a11 = A[i0,j0], a12 = A[i0,j1], a21 = A[i1,j0], a22 = A[i1,j1]
        b11 = B[j0,k0], b12 = B[j0,k1], b21 = B[j1,k0], b22 = B[j1,k1]
        c11 = C[i0,k0], c12 = C[i0,k1], c21 = C[i1,k0], c22 = C[i1,k1]
        
        W convention: W[k*n+i, c] so W[k0*i0_offset + i0], etc.
        Actually w_base already encodes the position in the p*n space.
        """
        # For each Strassen product, we need to combine U, V, W entries.
        # The trick: each Strassen "product" Mq is a product of a linear combo of A entries
        # and a linear combo of B entries. We represent this as a rank-1 term where
        # U has the linear combo coefficients for A, V has linear combo for B, and W
        # has coefficients for C.
        
        # But wait - in tensor decomposition, each rank-1 term is u_i * v_j * w_k.
        # For Strassen, M1 = (a11+a22)(b11+b22) contributes to c11, c22.
        # This means: U has [1,0,0,1] (for a11+a22), V has [1,0,0,1] (for b11+b22),
        # and W needs to have [1,0,0,1] (for c11 + c22 contribution).
        # But W entry is a single scalar per rank-1 term... 
        
        # Actually, in the tensor decomposition framework:
        # C[i,k] = sum_r U[i*m+j_r]... no wait.
        
        # Let me re-read the interface. U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Wait, that's not right either. Let me think about this more carefully.
        
        # The matrix multiplication tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] * B[j,k].
        # The tensor T has indices (i, j, k) and T[i,j,k] = 1 if we're computing the product.
        
        # Actually, the standard way: the multiplication tensor T_{m,n,p} is defined on 
        # V ⊗ V* ⊗ V* or similar. The rank-1 decomposition is:
        # T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
        
        # And the relation is: for all i,j,k: delta[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where delta[i,j,k] = 1 if (i,j,k) is a valid triple, 0 otherwise.
        
        # Wait, that's not quite right. The tensor for matrix multiplication is:
        # T(i, j, k) = 1 for all valid (i,j,k), and the decomposition gives:
        # The "structure constants" are such that C = AB can be computed as:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * ... 
        
        # I think the correct interpretation is:
        # For a rank-r decomposition, we compute r scalar products:
        # q_r = (sum_j U[i*m+j, r] * A[i,j]) ... no, that doesn't work dimensionally.
        
        # Let me reconsider. The standard tensor decomposition for matrix multiplication [n×m×p]:
        # We have tensors U (n*m × r), V (m*p × r), W (p*n × r).
        # The decomposition satisfies: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # and for all other (i,j,k) combinations that don't correspond to valid indices, the sum is 0.
        # But actually (i,j,k) always corresponds to a valid term in the product, so it's just = 1 for all valid triples.
        
        # Hmm, but that would mean the tensor is all-ones, which has rank 1. That can't be right.
        
        # I think the correct formulation is different. Let me look at this from the algorithm perspective:
        # A rank-r decomposition gives an algorithm:
        # 1. Compute r intermediate values: for r in 0..rank-1:
        #    s_r = sum_{i,j} U[i*m+j, r] * A[i,j]  -- NO, this sums over all i,j
        # 
        # Actually I think it's:
        # s_r = sum_j U_col_r[j] ... I'm confusing myself.
        
        # Let me just use the standard schoolbook and verify my understanding:
        # In schoolbook, rank = n*m*p.
        # For each (i,j,k): U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
        # And the identity is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for the matching (i,j,k)
        # and 0 otherwise. Since each r corresponds to exactly one (i',j',k'), 
        # U[i*m+j, r] = 1 iff i=i',j=j', similarly for V and W.
        # So the sum is 1 iff i=i' AND j=j' AND k=k', i.e., iff (i,j,k)=(i',j',k').
        # This equals the Kronecker delta, which is the "tensor" we're decomposing.
        
        # So the tensor T[i,j,k] = delta_{i,j,k} in the sense that it's 1 when the indices match a specific triple.
        # But that's just the identity tensor, which has rank n*m*p. That's consistent with schoolbook.
        
        # Wait, but Strassen has rank 7 < 8 for 2x2x2. So the tensor must be something else.
        
        # I think I'm misunderstanding. Let me reconsider.
        
        # The matrix multiplication tensor T_{m} for m×m×m is an element of R^{m*m} ⊗ R^{m*m} ⊗ R^{m*m}
        # defined by T[e_{ij}, e_{kl}, e_{lk}] = 1 if j=k, else 0. No wait...
        
        # Actually, the correct definition: T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} (flattened)
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k ⊗ ... 
        
        # I think the issue is that the tensor encodes the bilinear map (A,B) -> C.
        # The structure is: C[i,k] = sum_j A[i,j] B[j,k]
        # As a tensor: T[A_indices, B_indices, C_indices] = 1 if the indices form a valid product term.
        # Specifically, T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
        # But since (i*m+j, j*p+k, k*n+i) uniquely determines (i,j,k), this is a permutation-like tensor.
        
        # In the rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
        # This means: for all (a,b,c) where a=i*m+j, b=j'*p+k, c=k'*n+i':
        #   sum_r U[a,r] V[b,r] W[c,r] = 1 if a=i*m+j AND b=j*p+k AND c=k*n+i (consistent indices)
        #   and 0 otherwise.
        
        # "Consistent indices" means: from a=i*m+j we get i,j. From b=j'*p+k we get j',k. 
        # From c=k'*n+i' we get k',i'. Consistency means i=i', j=j', k=k'.
        # So T[a,b,c] = 1 iff the implied (i,j,k) triples are all equal.
        
        # This is a much more complex tensor than all-ones. And Strassen shows rank 7 for 2x2x2.
        
        # OK so for my Strassen decomposition, I need to construct U, V, W such that
        # the consistency condition is satisfied with only 7 terms instead of 8.
        
        # Let me look up how Strassen's 7 multiplications map to tensor entries.
        
        # For 2x2x2:
        # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]], C = [[c11,c12],[c21,c22]]
        # 
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # 
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # In tensor decomposition, each Mq corresponds to a rank-1 term:
        # u_q ⊗ v_q ⊗ w_q where:
        # u_q has entries for A coefficients, v_q for B coefficients, w_q for C coefficients.
        # 
        # M1: u1 = [1,0,0,1] (a11+a22), v1 = [1,0,0,1] (b11+b22), w1 = [1,0,0,1] (c11+c22)
        # M2: u2 = [0,0,1,1] (a21+a22), v2 = [1,0,0,0] (b11), w2 = [0,0,1,0] (c21)
        # M3: u3 = [1,0,0,0] (a11), v3 = [0,1,1,0] (b21-b22), w3 = [0,1,0,0] (c12)
        # M4: u4 = [0,0,0,1] (a22), v4 = [0,1,-1,0] (b12-b11), w4 = [1,0,1,0] (c11+c21)
        # M5: u5 = [1,1,0,0] (a11+a12), v5 = [0,0,0,1] (b22), w5 = [-1,0,0,1] (-c11+c22)
        # M6: u6 = [0,0,1,-1] (a21-a11), v6 = [1,1,0,0] (b11+b12), w6 = [0,0,0,1] (c22)
        # M7: u7 = [0,1,0,-1] (a12-a22), v7 = [0,0,1,1] (b21+b22), w7 = [1,0,0,0] (c11)
        
        # Let me verify M1: u1·A = a11+a22, v1·B = b11+b22, product = (a11+a22)(b11+b22) = M1.
        # w1 = [1,0,0,1] means M1 contributes +1 to c11 and +1 to c22. ✓
        
        # This looks correct! Now let me implement the full decomposition.
        
        pass  # placeholder, will implement properly below

    # I need to carefully build the U, V, W matrices.
    # Let me define helper functions.
    
    # For a 2x2x2 Strassen block at positions (i0, j0, k0) where:
    # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
    # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)  
    # C indices: (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)
    # W indices: (k0,i0), (k0,i0+1), (k0+1,i0), (k0+1,i0+1)
    
    def strassen_block(u_offsets, v_offsets, w_offsets):
        """
        Add 7 rank-1 terms for a 2x2x2 Strassen block.
        u_offsets: [u_00, u_01, u_10, u_11] - indices into U for A[i0,j0], A[i0,j0+1], A[i0+1,j0], A[i0+1,j0+1]
        v_offsets: [v_00, v_01, v_10, v_11] - indices into V for B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
        w_offsets: [w_00, w_01, w_10, w_11] - indices into W for C[i0,k0], C[i0,k0+1], C[i0+1,k0], C[i0+1,k0+1]
                   which correspond to W[k0*n+i0], W[k0*n+i0+1], W[(k0+1)*n+i0], W[(k0+1)*n+i0+1]
        """
        # 7 Strassen products
        # Each: (u_coeffs, v_coeffs, w_coeffs)
        terms = [
            # M1
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            # M2
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            # M3
            ([1,0,0,0], [0,1,1,0], [0,1,0,0]),
            # M4
            ([0,0,0,1], [0,1,-1,0], [1,0,1,0]),
            # M5
            ([1,1,0,0], [0,0,0,1], [-1,0,0,1]),
            # M6
            ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),
            # M7
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        for uc, vc, wc in terms:
            # Build this rank-1 term
            u_entry = [0] * (n * m)
            v_entry = [0] * (m * p)
            w_entry = [0] * (p * n)
            
            for idx, coeff in enumerate(uc):
                if coeff != 0:
                    u_entry[u_offsets[idx]] = coeff
            for idx, coeff in enumerate(vc):
                if coeff != 0:
                    v_entry[v_offsets[idx]] = coeff
            for idx, coeff in enumerate(wc):
                if coeff != 0:
                    w_entry[w_offsets[idx]] = coeff
            
            U_rows.append(u_entry)
            V_rows.append(v_entry)
            W_rows.append(w_entry)
    
    def schoolbook_block(u_offsets, v_offsets, w_offsets, size_u, size_v, size_w):
        """
        Add schoolbook rank-1 terms for a small block.
        u_offsets: list of U indices for the A sub-block (flattened row-major)
        v_offsets: list of V indices for the B sub-block
        w_offsets: list of W indices for the C sub-block
        """
        num_terms = len(u_offsets)
        for r in range(num_terms):
            u_entry = [0] * (n * m)
            v_entry = [0] * (m * p)
            w_entry = [0] * (p * n)
            u_entry[u_offsets[r]] = 1
            v_entry[v_offsets[r]] = 1
            w_entry[w_offsets[r]] = 1
            U_rows.append(u_entry)
            V_rows.append(v_entry)
            W_rows.append(w_entry)
    
    # Now, let's build the decomposition for 4x5x6.
    # Split m=5 into m1=3, m2=2.
    # Part 1: A is 4x3 (rows 0-3, cols 0-2 of A), B is 3x6 (rows 0-2 of B, all cols)
    # Part 2: A is 4x2 (rows 0-3, cols 3-4 of A), B is 2x6 (rows 3-4 of B, all cols)
    
    # For Part 1 (4x3x6):
    # Split n=4 into n1=2, n2=2. Split p=6 into p1=3, p2=3.
    # Four 2x3x3 blocks: (i0 in {0,2}, k0 in {0,3}), j in {0,1,2}
    
    # For each 2x3x3 block, decompose as:
    # 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) + 2x1x2 (schoolbook, rank 4) + 2x1x1 (schoolbook, rank 2)
    # Total rank per 2x3x3: 7+4+4+2 = 17
    
    # For Part 2 (4x2x6):
    # Split n=4 into 2+2, p=6 into 2+2+2.
    # Six 2x2x2 blocks, each Strassen rank 7. Total: 42.
    
    # Total rank: 4*17 + 6*7 = 68 + 42 = 110.
    
    # Let me define index helper functions.
    def u_idx(i, j):
        return i * m + j  # index into U for A[i,j]
    
    def v_idx(j, k):
        return j * p + k  # index into V for B[j,k]
    
    def w_idx(k, i):
        return k * n + i  # index into W for C[i,k] (W[k*n+i])
    
    # Decompose a 2x3x3 block:
    # A is 2x3: rows i0, i0+1; cols j0, j0+1, j0+2
    # B is 3x3: rows j0, j0+1, j0+2; cols k0, k0+1, k0+2
    # C is 2x3: rows i0, i0+1; cols k0, k0+1, k0+2
    
    def decomp_2x3x3(i0, j0, k0):
        """
        Decompose 2x3x3 block starting at (i0, j0, k0).
        A: rows i0, i0+1; cols j0, j0+1, j0+2
        B: rows j0, j0+1, j0+2; cols k0, k0+1, k0+2
        C: rows i0, i0+1; cols k0, k0+1, k0+2
        """
        # Sub-block 1: 2x2x2 using A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[i0:i0+2, k0:k0+2]
        u_222 = [u_idx(i0, j0), u_idx(i0, j0+1), u_idx(i0+1, j0), u_idx(i0+1, j0+1)]
        v_222 = [v_idx(j0, k0), v_idx(j0, k0+1), v_idx(j0+1, k0), v_idx(j0+1, k0+1)]
        w_222 = [w_idx(k0, i0), w_idx(k0, i0+1), w_idx(k0+1, i0), w_idx(k0+1, i0+1)]
        strassen_block(u_222, v_222, w_222)
        
        # Sub-block 2: 2x2x1 using A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0+2], C[i0:i0+2, k0+2]
        # This is just 2x2 products with one column. Schoolbook rank 4.
        for di in range(2):
            for dj in range(2):
                u_entry = [0] * (n * m)
                v_entry = [0] * (m * p)
                w_entry = [0] * (p * n)
                u_entry[u_idx(i0+di, j0+dj)] = 1
                v_entry[v_idx(j0+dj, k0+2)] = 1
                w_entry[w_idx(k0+2, i0+di)] = 1
                U_rows.append(u_entry)
                V_rows.append(v_entry)
                W_rows.append(w_entry)
        
        # Sub-block 3: 2x1x2 using A[i0:i0+2, j0+2], B[j0+2, k0:k0+2], C[i0:i0+2, k0:k0+2]
        # Schoolbook rank 4.
        for di in range(2):
            for dk in range(2):
                u_entry = [0] * (n * m)
                v_entry = [0] * (m * p)
                w_entry = [0] * (p * n)
                u_entry[u_idx(i0+di, j0+2)] = 1
                v_entry[v_idx(j0+2, k0+dk)] = 1
                w_entry[w_idx(k0+dk, i0+di)] = 1
                U_rows.append(u_entry)
                V_rows.append(v_entry)
                W_rows.append(w_entry)
        
        # Sub-block 4: 2x1x1 using A[i0:i0+2, j0+2], B[j0+2, k0+2], C[i0:i0+2, k0+2]
        # Schoolbook rank 2.
        for di in range(2):
            u_entry = [0] * (n * m)
            v_entry = [0] * (m * p)
            w_entry = [0] * (p * n)
            u_entry[u_idx(i0+di, j0+2)] = 1
            v_entry[v_idx(j0+2, k0+2)] = 1
            w_entry[w_idx(k0+2, i0+di)] = 1
            U_rows.append(u_entry)
            V_rows.append(v_entry)
            W_rows.append(w_entry)
    
    # Part 1: 4x3x6
    # Four 2x3x3 blocks: (i0, k0) in {(0,0), (0,3), (2,0), (2,3)}, j0=0
    for i0 in [0, 2]:
        for k0 in [0, 3]:
            decomp_2x3x3(i0, 0, k0)
    
    # Part 2: 4x2x6
    # A is 4x2: rows 0-3, cols 3-4 of A (i.e., j in {3,4})
    # B is 2x6: rows 3-4 of B (j in {3,4}), cols 0-5
    # Split n=4 into 2+2, p=6 into 2+2+2.
    # Six 2x2x2 blocks.
    for i0 in [0, 2]:
        for k0 in [0, 2, 4]:
            u_222 = [u_idx(i0, 3), u_idx(i0, 4), u_idx(i0+1, 3), u_idx(i0+1, 4)]
            v_222 = [v_idx(3, k0), v_idx(3, k0+1), v_idx(4, k0), v_idx(4, k0+1)]
            w_222 = [w_idx(k0, i0), w_idx(k0, i0+1), w_idx(k0+1, i0), w_idx(k0+1, i0+1)]
            strassen_block(u_222, v_222, w_222)
    
    # Convert to proper format
    rank = len(U_rows)
    U = U_rows  # shape (rank, n*m) - but we need (n*m, rank)
    V = V_rows  # shape (rank, m*p) - but we need (m*p, rank)
    W = W_rows  # shape (rank, p*n) - but we need (p*n, rank)
    
    # Transpose
    U_final = [[U[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W[r][i] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
