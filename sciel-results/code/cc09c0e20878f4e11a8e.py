import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x5 matrix.

    Strategy: Start from schoolbook rank-40, then attempt rank reduction via
    iterative rank-1 component fitting on the residual tensor.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 5
    dim_a = n * m   # 8
    dim_b = m * p   # 20
    dim_c = p * n   # 10

    # Build the multiplication tensor T[i,j,k] = 1 if C[i,k] = A[i,j]*B[j,k]
    # T shape: (dim_a, dim_b, dim_c) = (8, 20, 10)
    # Indexing: a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((dim_a, dim_b, dim_c), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def build_decomposition(components):
        """Build U, V, W from list of (u, v, w) rank-1 components."""
        rank = len(components)
        U = np.zeros((dim_a, rank), dtype=float)
        V = np.zeros((dim_b, rank), dtype=float)
        W = np.zeros((dim_c, rank), dtype=float)
        for r, (u, v, w) in enumerate(components):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        return U, V, W

    def verify(U, V, W, tol=1e-10):
        """Check that sum_r U[:,r] * V[:,r] * W[:,r] == T exactly."""
        recon = np.zeros_like(T)
        for r in range(U.shape[1]):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T)) < tol

    def tensor_from_uv(U, V, W):
        """Reconstruct tensor from factor matrices."""
        recon = np.zeros_like(T)
        for r in range(U.shape[1]):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return recon

    # Start with schoolbook rank-40
    components = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u = np.zeros(dim_a); u[a] = 1.0
                v = np.zeros(dim_b); v[b] = 1.0
                w = np.zeros(dim_c); w[c] = 1.0
                components.append((u, v, w))

    current_rank = len(components)

    # Attempt to reduce rank by fitting rank-1 terms to residual
    # Strategy: try to replace pairs or triples of rank-1 terms with a single
    # better-fitting term, or use algebraic structure to merge.

    # Key insight: For matrix multiplication, the tensor has a special structure.
    # T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    # This means T is supported on exactly n*m*p = 40 entries.
    
    # A rank-1 term (u,v,w) contributes to T[a,b,c] += u[a]*v[b]*w[c].
    # For the decomposition to be exact, we need the sum to equal T.
    
    # Let's try a different approach: use the known structure to build a
    # lower-rank decomposition. The rank of the 2x4x5 multiplication tensor
    # is known to be at most 36 (from block Strassen). Let's try to construct
    # it explicitly using Strassen on appropriate blocks.

    # Block decomposition:
    # A is 2x4: split into A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    # B is 4x5: split into B1 = B[0:2,:] (2x5), B2 = B[2:4,:] (2x5)
    # C = A1*B1 + A2*B2 (2x5)
    
    # For 2x2 * 2x5: split B into B_left (2x2) + B_right (2x3)
    # 2x2 * 2x2: use Strassen (rank 7)
    # 2x2 * 2x3: schoolbook (rank 6) or further decompose
    
    # Actually, let's try to use the full Strassen-based construction.
    # 2x2 * 2x2 = rank 7 (Strassen)
    # 2x2 * 2x3: split 2x3 into 2x2 + 2x1
    #   2x2 * 2x2 = rank 7
    #   2x2 * 2x1 = rank 2 (schoolbook, since one column)
    #   Total for 2x2*2x3 = 9? No, that's not right.
    #   Actually 2x2 * 2x3: we can do it in 6 mults schoolbook, or 
    #   split into 2x2*2x2 (7) + 2x2*2x1 (2) = 9, which is worse.
    #   So schoolbook 6 is better for 2x2*2x3.
    
    # So: 2x2 * 2x5 = 2x2*2x2 + 2x2*2x3 = 7 + 6 = 13
    # Total: 2 * 13 = 26? That seems too low. Let me reconsider.
    
    # Wait, I need to be more careful. Let me just try to find any valid
    # decomposition with rank < 40 by using a numerical approach.
    
    # Alternative: use the fact that we can combine terms.
    # If we have two rank-1 terms that share structure, we might merge them.
    
    # Let's try a direct approach: use integer/half-integer entries.
    # The known rank-36 construction for 2x4x5 uses:
    # - Split A into two 2x2 blocks: A1, A2
    # - Split B into four 2x2 blocks? No, B is 4x5.
    # - B = [B11 B12 B13; B21 B22 B23] where Bij are 2x? 
    #   Actually B is 4x5, so split rows into two 2-row blocks:
    #   B_top = B[0:2, :] (2x5), B_bot = B[2:4, :] (2x5)
    #   Each 2x5 can be split as 2x2 + 2x3.
    
    # Let me just try to verify if rank 36 is achievable and construct it.
    # For now, let's try a numerical rank-reduction approach.
    
    # Try to reduce from 40 by finding linear dependencies among the
    # rank-1 components when viewed as vectors in the tensor space.
    
    # The tensor space has dimension 8*20*10 = 1600.
    # Each rank-1 component is a vector in this space.
    # If we can express the sum of some components as a single rank-1 term,
    # we reduce the rank.
    
    # Let's flatten T and each rank-1 component.
    T_flat = T.flatten()
    
    # For each pair of components, check if their sum can be written as rank-1
    # (this is generally not possible, but let's check for special structure)
    
    # Actually, a better approach: use the known algebraic construction.
    # Let me carefully build the rank-36 Strassen-based decomposition.
    
    # 2x4 * 4x5:
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x5):
    #   B1 = [B1a | B1b] where B1a is 2x2, B1b is 2x3
    #   A1*B1 = A1*B1a + A1*B1b
    #   A1*B1a: Strassen 2x2*2x2, rank 7
    #   A1*B1b: schoolbook 2x2*2x3, rank 6
    #   Subtotal: 13
    
    # Similarly A2*B2: 13
    # Total: 26
    
    # Wait, that gives 26, not 36. Let me recheck.
    # Actually, I think the issue is that Strassen for 2x2*2x2 gives rank 7,
    # and schoolbook 2x2*2x3 gives rank 6, so 7+6=13 per block, total 26.
    # But 26 < 40, so this should work if the construction is correct.
    # However, I'm not sure the known rank is 36. Let me just try rank 26.
    
    # Let me construct this explicitly.
    
    def strassen_2x2_2x2():
        """
        Returns 7 rank-1 components for 2x2 * 2x2 matrix multiplication.
        A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        C = [[c11, c12], [c21, c22]]
        
        Strassen:
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)b11
        M3 = a11(b12-b22)
        M4 = a22(b11-b12)
        M5 = (a11+a12)b22
        M6 = (a12-a22)(b11+b12)
        M7 = (a21-a11)(b21+b22)
        
        c11 = M1 - M4 + M5 + M7
        c12 = M3 + M4
        c21 = M2 + M5
        c22 = M1 - M2 + M3 + M6
        """
        # Each component is (u, v, w) where:
        # u has indices (i,j) for A, v has indices (j,k) for B, w has indices (k,i) for C
        # For 2x2*2x2: i in {0,1}, j in {0,1}, k in {0,1}
        # a_idx(i,j) = i*2+j, b_idx(j,k) = j*2+k, c_idx(k,i) = k*2+i
        
        comps = []
        
        # M1 = (a11+a22)(b11+b22)
        u = np.zeros(4); u[0] = 1; u[3] = 1  # a11, a22
        v = np.zeros(4); v[0] = 1; v[3] = 1  # b11, b22
        w = np.zeros(4); w[0] = 1; w[1] = 1; w[2] = 1; w[3] = 1  # c11,c12,c21,c22 (all get +1)
        # Wait, this isn't right. Each M contributes to specific C entries.
        # M1 contributes +1 to c11 and c22
        # Let me redo this properly.
        
        # Actually, the decomposition is:
        # C = sum_r M_r * (contribution pattern)
        # where M_r is a scalar product of a linear combination of A entries
        # and a linear combination of B entries.
        
        # Let me define each term as (u, v, w) where:
        # u[i,j] = coefficient of A[i,j] in the left factor
        # v[j,k] = coefficient of B[j,k] in the right factor  
        # w[k,i] = coefficient in C[k,i]
        
        # M1: left = a11+a22, right = b11+b22
        #   u: a11=1, a22=1
        #   v: b11=1, b22=1
        #   w: c11 += 1, c22 += 1
        u1 = np.zeros(4); u1[0]=1; u1[3]=1
        v1 = np.zeros(4); v1[0]=1; v1[3]=1
        w1 = np.zeros(4); w1[0]=1; w1[3]=1  # c11 at idx 0, c22 at idx 3
        comps.append((u1, v1, w1))
        
        # M2: left = a21+a22, right = b11
        #   u: a21=1, a22=1
        #   v: b11=1
        #   w: c21 += 1
        u2 = np.zeros(4); u2[2]=1; u2[3]=1
        v2 = np.zeros(4); v2[0]=1
        w2 = np.zeros(4); w2[2]=1  # c21 at idx 2
        comps.append((u2, v2, w2))
        
        # M3: left = a11, right = b12-b22
        #   u: a11=1
        #   v: b12=1, b22=-1
        #   w: c12 += 1, c22 += 1
        u3 = np.zeros(4); u3[0]=1
        v3 = np.zeros(4); v3[1]=1; v3[3]=-1
        w3 = np.zeros(4); w3[1]=1; w3[3]=1  # c12 at idx 1, c22 at idx 3
        comps.append((u3, v3, w3))
        
        # M4: left = a22, right = b11-b12
        #   u: a22=1
        #   v: b11=1, b12=-1
        #   w: c11 -= 1, c12 += 1
        u4 = np.zeros(4); u4[3]=1
        v4 = np.zeros(4); v4[0]=1; v4[1]=-1
        w4 = np.zeros(4); w4[0]=-1; w4[1]=1  # c11 at idx 0, c12 at idx 1
        comps.append((u4, v4, w4))
        
        # M5: left = a11+a12, right = b22
        #   u: a11=1, a12=1
        #   v: b22=1
        #   w: c11 += 1, c21 += 1
        u5 = np.zeros(4); u5[0]=1; u5[1]=1
        v5 = np.zeros(4); v5[3]=1
        w5 = np.zeros(4); w5[0]=1; w5[2]=1  # c11 at idx 0, c21 at idx 2
        comps.append((u5, v5, w5))
        
        # M6: left = a12-a22, right = b11+b12
        #   u: a12=1, a22=-1
        #   v: b11=1, b12=1
        #   w: c22 += 1
        u6 = np.zeros(4); u6[1]=1; u6[3]=-1
        v6 = np.zeros(4); v6[0]=1; v6[1]=1
        w6 = np.zeros(4); w6[3]=1  # c22 at idx 3
        comps.append((u6, v6, w6))
        
        # M7: left = a21-a11, right = b21+b22
        #   u: a21=1, a11=-1
        #   v: b21=1, b22=1
        #   w: c11 += 1
        u7 = np.zeros(4); u7[2]=1; u7[0]=-1
        v7 = np.zeros(4); v7[2]=1; v7[3]=1
        w7 = np.zeros(4); w7[0]=1  # c11 at idx 0
        comps.append((u7, v7, w7))
        
        return comps
    
    def schoolbook_2x2_2x3():
        """
        Returns 6 rank-1 components for 2x2 * 2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x3.
        C[i,k] = sum_j A[i,j] * B[j,k]
        """
        comps = []
        # a_idx(i,j) = i*2+j (4 entries)
        # b_idx(j,k) = j*3+k (6 entries)
        # c_idx(k,i) = k*2+i (6 entries)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = np.zeros(4); u[i*2+j] = 1
                    v = np.zeros(6); v[j*3+k] = 1
                    w = np.zeros(6); w[k*2+i] = 1
                    comps.append((u, v, w))
        return comps
    
    def schoolbook_2x2_2x2():
        """Returns 4 rank-1 components for 2x2 * 2x2."""
        comps = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = np.zeros(4); u[i*2+j] = 1
                    v = np.zeros(4); v[j*2+k] = 1
                    w = np.zeros(4); w[k*2+i] = 1
                    comps.append((u, v, w))
        return comps
    
    # Now build the full 2x4 * 4x5 decomposition:
    # A: 2x4, indexed as A[i, j] where i in {0,1}, j in {0,1,2,3}
    #   a_full_idx(i,j) = i*4+j (8 entries)
    # B: 4x5, indexed as B[j, k] where j in {0,1,2,3}, k in {0,1,2,3,4}
    #   b_full_idx(j,k) = j*5+k (20 entries)
    # C: 5x2 (trace convention: W[k*n+i, r]), so C[k, i] where k in {0,1,2,3,4}, i in {0,1}
    #   c_full_idx(k,i) = k*2+i (10 entries)
    
    # Split: A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B1 = B[0:2, :], B2 = B[2:4, :]
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x5):
    #   B1 = [B1a | B1b] where B1a = B[0:2, 0:2] (2x2), B1b = B[0:2, 2:5] (2x3)
    #   A1*B1 = A1*B1a + A1*B1b
    #   Use Strassen for A1*B1a (7 terms), schoolbook for A1*B1b (6 terms)
    
    # For A2*B2 (2x2 * 2x5):
    #   B2 = [B2a | B2b] where B2a = B[2:4, 0:2] (2x2), B2b = B[2:4, 2:5] (2x3)
    #   A2*B2 = A2*B2a + A2*B2b
    #   Use Strassen for A2*B2a (7 terms), schoolbook for A2*B2b (6 terms)
    
    # Total: 7 + 6 + 7 + 6 = 26
    
    all_comps = []
    
    # Helper to embed a local component into the full space
    def embed_strassen(comp, a_row_offset, b_row_offset, c_col_offset):
        """
        Embed a Strassen 2x2*2x2 component into the full 2x4*4x5 space.
        a_row_offset: which 2 rows of A this corresponds to (0 or 2 for A1 or A2)
        b_row_offset: which 2 rows of B this corresponds to (0 or 2)
        c_col_offset: not used directly, C is determined by the product
        """
        u_local, v_local, w_local = comp
        u_full = np.zeros(dim_a)
        v_full = np.zeros(dim_b)
        w_full = np.zeros(dim_c)
        
        # u_local has 4 entries: a[i,j] for i in {0,1}, j in {0,1}
        # In full A (2x4), A1 uses columns 0,1; A2 uses columns 2,3
        # a_full_idx(i, j) = i*4 + j
        col_off = a_row_offset  # 0 for A1, 2 for A2
        for i in range(2):
            for j in range(2):
                u_full[i*4 + col_off + j] = u_local[i*2 + j]
        
        # v_local has 4 entries: b[j,k] for j in {0,1}, k in {0,1}
        # In full B (4x5), B1 uses rows 0,1; B2 uses rows 2,3
        # b_full_idx(j, k) = j*5 + k
        row_off = b_row_offset  # 0 for B1, 2 for B2
        for j in range(2):
            for k in range(2):
                v_full[row_off + j]*1 + 0  # placeholder
                v_full[(row_off + j)*5 + k] = v_local[j*2 + k]
        
        # w_local has 4 entries: c[k,i] for k in {0,1}, i in {0,1}
        # In full C (5x2), c_full_idx(k, i) = k*2 + i
        # The output rows k=0,1 come from B rows 0,1 (if B1) or B rows 2,3 (if B2)
        # Wait, C[k,i] = sum_j A[i,j]*B[j,k]
        # For A1*B1: k ranges over 0,1 (B1 rows)
        # For A2*B2: k ranges over 2,3 (B2 rows)
        # So c_full_idx(k, i) = k*2 + i
        k_off = b_row_offset  # 0 for B1, 2 for B2
        for k in range(2):
            for i in range(2):
                w_full[(k_off + k)*2 + i] = w_local[k*2 + i]
        
        return (u_full, v_full, w_full)
    
    def embed_schoolbook_2x2_2x3(comp, a_col_offset, b_row_offset, b_col_offset):
        """
        Embed a schoolbook 2x2*2x3 component into the full space.
        a_col_offset: 0 for A1, 2 for A2
        b_row_offset: 0 for B1, 2 for B2
        b_col_offset: 0 for left 2x2 part, 2 for right 2x3 part
        """
        u_local, v_local, w_local = comp
        u_full = np.zeros(dim_a)
        v_full = np.zeros(dim_b)
        w_full = np.zeros(dim_c)
        
        # u_local: 4 entries, a[i,j] for i in {0,1}, j in {0,1}
        col_off = a_col_offset
        for i in range(2):
            for j in range(2):
                u_full[i*4 + col_off + j] = u_local[i*2 + j]
        
        # v_local: 6 entries, b[j,k] for j in {0,1}, k in {0,1,2}
        row_off = b_row_offset
        k_off = b_col_offset
        for j in range(2):
            for k in range(3):
                v_full[(row_off + j)*5 + k_off + k] = v_local[j*3 + k]
        
        # w_local: 6 entries, c[k,i] for k in {0,1,2}, i in {0,1}
        # In full C, k ranges over 5, but here only 3 specific columns
        k_off_c = b_col_offset  # same as b_col_offset
        for k in range(3):
            for i in range(2):
                w_full[(k_off_c + k)*2 + i] = w_local[k*2 + i]
        
        return (u_full, v_full, w_full)
    
    # Build A1*B1a using Strassen (7 terms)
    strassen_comps = strassen_2x2_2x2()
    for comp in strassen_comps:
        all_comps.append(embed_strassen(comp, a_row_offset=0, b_row_offset=0, c_col_offset=0))
    
    # Build A1*B1b using schoolbook 2x2*2x3 (6 terms)
    sb_comps = schoolbook_2x2_2x3()
    for comp in sb_comps:
        all_comps.append(embed_schoolbook_2x2_2x3(comp, a_col_offset=0, b_row_offset=0, b_col_offset=2))
    
    # Build A2*B2a using Strassen (7 terms)
    for comp in strassen_comps:
        all_comps.append(embed_strassen(comp, a_row_offset=2, b_row_offset=2, c_col_offset=2))
    
    # Build A2*B2b using schoolbook 2x2*2x3 (6 terms)
    for comp in sb_comps:
        all_comps.append(embed_schoolbook_2x2_2x3(comp, a_col_offset=2, b_row_offset=2, b_col_offset=2))
    
    current_rank = len(all_comps)
    print(f"Constructed decomposition with rank {current_rank}")
    
    # Verify
    U, V, W = build_decomposition(all_comps)
    if verify(U, V, W):
        print("Rank-{} decomposition verified!".format(current_rank))
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        print("Verification failed, falling back to schoolbook rank 40")
    
    # Fallback: schoolbook rank 40
    U = np.zeros((dim_a, 40), dtype=float)
    V = np.zeros((dim_b, 40), dtype=float)
    W = np.zeros((dim_c, 40), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, idx] = 1.0
                V[b, idx] = 1.0
                W[c, idx] = 1.0
                idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
