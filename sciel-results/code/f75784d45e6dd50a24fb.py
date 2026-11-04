<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_val, v_val, w_val):
        U_list.append(u_val)
        V_list.append(v_val)
        W_list.append(w_val)
    
    def add_schoolbook(i, j, k):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        add_term(u, v, w)
    
    def add_strassen(i0, j0, k0):
        # Strassen decomposition for 2x2x2 starting at (i0, j0, k0)
        # A = [[a,b],[c,d]] = A[i0:i0+2, j0:j0+2]
        # B = [[e,f],[g,h]] = B[j0:j0+2, k0:k0+2]
        # C = [[p,q],[r,s]] = C[i0:i0+2, k0:k0+2]
        
        a = (i0, j0)
        b = (i0, j0+1)
        c = (i0+1, j0)
        d = (i0+1, j0+1)
        e = (j0, k0)
        f = (j0, k0+1)
        g = (j0+1, k0)
        h = (j0+1, k0+1)
        
        # M1 = (a+d)(e+h)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[a[0]*m+a[1]] = 1; u1[d[0]*m+d[1]] = 1
        v1[e[0]*p+e[1]] = 1; v1[h[0]*p+h[1]] = 1
        w1[(k0)*n+i0] = 1; w1[(k0)*n+i0+1] = 1
        w1[(k0+1)*n+i0] = 1; w1[(k0+1)*n+i0+1] = 1
        # Wait, this is wrong. Strassen computes products, not sums.
        
        # I'm confusing myself. Let me look up the correct Strassen decomposition.
        
        # Strassen: 
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M is a product of a linear form in A and a linear form in B.
        # So each M corresponds to one rank-1 term in the tensor decomposition.
        
        # For M1: U has 1 at (i0,j0) and (i0+1,j0+1), V has 1 at (j0,k0) and (j0+1,k0+1)
        # But W needs to distribute to the 4 outputs.
        
        # Actually, in the tensor decomposition, each rank-1 term is:
        # U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # For M1 = (A11+A22)*(B11+B22), this contributes to all 4 outputs of C.
        # Specifically:
        # C11 += M1, C12 += 0, C21 += 0, C22 += M1 (from the formulas above, M1 appears in C11 and C22)
        
        # Wait, from the formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes +1 to C11 and +1 to C22.
        
        # In the tensor decomposition, for the term corresponding to M1:
        # U[i,j] = 1 if (i,j) in {(i0,j0), (i0+1,j0+1)}
        # V[j,k] = 1 if (j,k) in {(j0,k0), (j0+1,k0+1)}
        # W[k,i] = 1 if (k,i) in {(k0,i0), (k0+1,i0+1)}  [for C11 and C22]
        # But W also needs to be 0 for C12 and C21.
        
        # This is not a single rank-1 term! M1 contributes to two different outputs.
        
        # I think I'm misunderstanding the tensor decomposition.
        
        # In the tensor decomposition, each rank-1 term is a product of three vectors.
        # The Strassen decomposition has 7 such terms, each contributing to multiple outputs.
        
        # Let me define each of the 7 terms explicitly.
        
        # Term 1 (M1): U1[i,j] = 1 if (i,j) = (i0,j0) or (i0+1,j0+1)
        #              V1[j,k] = 1 if (j,k) = (j0,k0) or (j0+1,k0+1)
        #              W1[k,i] = 1 if (k,i) = (k0,i0) or (k0+1,i0+1)
        # This gives: sum over (i,j) in {(i0,j0),(i0+1,j0+1)} of sum over (j,k) in {(j0,k0),(j0+1,k0+1)} of A[i,j]*B[j,k] * W1[k,i]
        # = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1]) * (W1[k0,i0] + W1[k0+1,i0+1] + cross terms)
        
        # This is getting too messy. Let me use a different approach.
        
        # I'll just use schoolbook for now and accept rank 84.
        pass
    
    # Just use schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Use Strassen-style decomposition where possible to reduce rank
    # For 3x4 * 4x7, we can partition into blocks and use Strassen on 2x2 blocks
    
    # Strategy: Partition A (3x4) and B (4x7) into blocks
    # We'll use a combination of schoolbook and Strassen-like reductions
    
    # For now, let's implement a proper Strassen decomposition for 2x2x2 blocks
    # and use schoolbook for the remaining parts
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    def add_term(u_sparse, v_sparse, w_sparse):
        """Add a rank-1 term with sparse representations.
        u_sparse: list of (index, value)
        v_sparse: list of (index, value)
        w_sparse: list of (index, value)
        """
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in u_sparse:
            u[idx] += val
        for idx, val in v_sparse:
            v[idx] += val
        for idx, val in w_sparse:
            w[idx] += val
        terms_U.append(u)
        terms_V.append(v)
        terms_W.append(w)
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # Strassen decomposition for a 2x2 block starting at (i0, j0, k0)
    # This handles A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    def add_strassen_2x2(i0, j0, k0):
        # M1 = (A11 + A22) * (B11 + B22)
        # Contributes to C11 and C22
        u1 = [(u_idx(i0, j0), 1), (u_idx(i0+1, j0+1), 1)]
        v1 = [(v_idx(j0, k0), 1), (v_idx(j0+1, k0+1), 1)]
        w1 = [(w_idx(k0, i0), 1), (w_idx(k0+1, i0+1), 1)]
        add_term(u1, v1, w1)
        
        # M2 = (A21 + A22) * B11
        # Contributes to C21
        u2 = [(u_idx(i0+1, j0), 1), (u_idx(i0+1, j0+1), 1)]
        v2 = [(v_idx(j0, k0), 1)]
        w2 = [(w_idx(k0, i0+1), 1)]
        add_term(u2, v2, w2)
        
        # M3 = A11 * (B12 - B22)
        # Contributes to C12
        u3 = [(u_idx(i0, j0), 1)]
        v3 = [(v_idx(j0, k0+1), 1), (v_idx(j0+1, k0+1), -1)]
        w3 = [(w_idx(k0+1, i0), 1)]
        add_term(u3, v3, w3)
        
        # M4 = A22 * (B21 - B11)
        # Contributes to C11 and C21
        u4 = [(u_idx(i0+1, j0+1), 1)]
        v4 = [(v_idx(j0+1, k0), 1), (v_idx(j0, k0), -1)]
        w4 = [(w_idx(k0, i0), 1), (w_idx(k0, i0+1), 1)]
        # Wait, this doesn't work as a single rank-1 term because W has two entries
        # Let me reconsider...
        
        # Actually, in tensor decomposition, each rank-1 term must have exactly one non-zero
        # entry in W for each (k,i) pair it affects. But if W has multiple non-zero entries,
        # it's still valid as long as the product works out.
        
        # Let me verify: The contribution is sum_{i,j,k} U[i,j] * V[j,k] * W[k,i] * A[i,j] * B[j,k]
        # For M4: U has A22, V has (B21 - B11), W has entries at (k0, i0) and (k0, i0+1)
        # This gives: A22 * (B21 - B11) * [W[k0,i0] * C[k0,i0] + W[k0,i0+1] * C[k0,i0+1]]
        # = A22 * (B21 - B11) * (C21_part + C22_part) if W[k0,i0]=1 and W[k0,i0+1]=1
        # But we want M4 to contribute to C11 and C21, not C22.
        
        # I think I need to be more careful. Let me use the standard Strassen formulation
        # where each M_r is a scalar product, and then C is a linear combination of M_r's.
        
        # In the tensor decomposition framework:
        # C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] summed over j and r
        # More precisely: C[i,k] = sum_{j,r} U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
        
        # So for each rank-1 term r, we have vectors U_r (over i,j), V_r (over j,k), W_r (over k,i)
        # and the contribution is (U_r · A_flat) * (V_r · B_flat) * W_r[k,i] for each (k,i)
        
        # This means W_r can have multiple non-zero entries, and each contributes to a different output.
        
        # Let me redo this properly.
        
        # For M1 = (A11+A22)*(B11+B22):
        # U1: 1 at (i0,j0) and (i0+1,j0+1)
        # V1: 1 at (j0,k0) and (j0+1,k0+1)
        # W1: 1 at (k0,i0) and (k0+1,i0+1)
        # Contribution: (A11+A22)(B11+B22) * [C[k0,i0] + C[k0+1,i0+1] + cross terms]
        # But cross terms: A11*B[j0+1,k0+1] etc. are not what we want.
        
        # The issue is that the tensor product U⊗V⊗W creates all combinations.
        # So (A11+A22)*(B11+B22) expands to A11*B11 + A11*B22 + A22*B11 + A22*B22
        # but we only want A11*B11 + A22*B22 for M1... no wait, M1 IS (A11+A22)*(B11+B22)
        # which does include all four cross terms. That's the point of Strassen - we compute
        # these products and then combine them linearly.
        
        # So the W vector for M1 should have 1 at (k0,i0) and (k0+1,i0+1) because
        # M1 contributes to both C11 and C22.
        
        # But wait, the cross terms A11*B22 and A22*B11 also get multiplied by W.
        # So they also contribute to C11 and C22. Is that correct?
        
        # In Strassen:
        # M1 = (A11+A22)*(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M2 + M3 + M6
        # So yes, M1 contributes to both C11 and C22, and the cross terms are part of M1.
        
        # So the decomposition is correct! Each M_r is a rank-1 term in the tensor sense.
        
        # Let me continue with the other terms.
        
        # M2 = (A21+A22)*B11
        # U2: 1 at (i0+1,j0) and (i0+1,j0+1)
        # V2: 1 at (j0,k0)
        # W2: 1 at (k0,i0+1)  [M2 contributes to C21]
        # But wait, C21 = M2 + M4, so M2 contributes to C21 only.
        
        # M3 = A11*(B12-B22)
        # U3: 1 at (i0,j0)
        # V3: 1 at (j0,k0+1), -1 at (j0+1,k0+1)
        # W3: 1 at (k0+1,i0)  [M3 contributes to C12]
        
        # M4 = A22*(B21-B11)
        # U4: 1 at (i0+1,j0+1)
        # V4: 1 at (j0+1,k0), -1 at (j0,k0)
        # W4: 1 at (k0,i0) and 1 at (k0,i0+1)  [M4 contributes to C11 and C21]
        
        # M5 = (A11+A12)*B22
        # U5: 1 at (i0,j0) and (i0,j0+1)
        # V5: 1 at (j0+1,k0+1)
        # W5: -1 at (k0,i0) and 1 at (k0+1,i0)  [M5 contributes to -C11 and C12]
        
        # M6 = (A21-A11)*(B11+B12)
        # U6: 1 at (i0+1,j0), -1 at (i0,j0)
        # V6: 1 at (j0,k0) and 1 at (j0,k0+1)
        # W6: 1 at (k0+1,i0+1)  [M6 contributes to C22]
        
        # M7 = (A12-A22)*(B21+B22)
        # U7: 1 at (i0,j0+1), -1 at (i0+1,j0+1)
        # V7: 1 at (j0+1,k0) and 1 at (j0+1,k0+1)
        # W7: 1 at (k0,i0)  [M7 contributes to C11]
        
        # Let me add all these terms.
        
        # M1
        u1 = [(u_idx(i0, j0), 1), (u_idx(i0+1, j0+1), 1)]
        v1 = [(v_idx(j0, k0), 1), (v_idx(j0+1, k0+1), 1)]
        w1 = [(w_idx(k0, i0), 1), (w_idx(k0+1, i0+1), 1)]
        add_term(u1, v1, w1)
        
        # M2
        u2 = [(u_idx(i0+1, j0), 1), (u_idx(i0+1, j0+1), 1)]
        v2 = [(v_idx(j0, k0), 1)]
        w2 = [(w_idx(k0, i0+1), 1)]
        add_term(u2, v2, w2)
        
        # M3
        u3 = [(u_idx(i0, j0), 1)]
        v3 = [(v_idx(j0, k0+1), 1), (v_idx(j0+1, k0+1), -1)]
        w3 = [(w_idx(k0+1, i0), 1)]
        add_term(u3, v3, w3)
        
        # M4
        u4 = [(u_idx(i0+1, j0+1), 1)]
        v4 = [(v_idx(j0+1, k0), 1), (v_idx(j0, k0), -1)]
        w4 = [(w_idx(k0, i0), 1), (w_idx(k0, i0+1), 1)]
        add_term(u4, v4, w4)
        
        # M5
        u5 = [(u_idx(i0, j0), 1), (u_idx(i0, j0+1), 1)]
        v5 = [(v_idx(j0+1, k0+1), 1)]
        w5 = [(w_idx(k0, i0), -1), (w_idx(k0+1, i0), 1)]
        add_term(u5, v5, w5)
        
        # M6
        u6 = [(u_idx(i0+1, j0), 1), (u_idx(i0, j0), -1)]
        v6 = [(v_idx(j0, k0), 1), (v_idx(j0, k0+1), 1)]
        w6 = [(w_idx(k0+1, i0+1), 1)]
        add_term(u6, v6, w6)
        
        # M7
        u7 = [(u_idx(i0, j0+1), 1), (u_idx(i0+1, j0+1), -1)]
        v7 = [(v_idx(j0+1, k0), 1), (v_idx(j0+1, k0+1), 1)]
        w7 = [(w_idx(k0, i0), 1)]
        add_term(u7, v7, w7)
    
    # For 3x4 * 4x7:
    # We can partition A into rows [0:2] and [2:3], and columns [0:2], [2:3], [3:4]
    # And B into rows [0:2], [2:3], [3:4] and columns [0:2], [2:4], [4:7]
    
    # This is getting complex. Let's use a simpler approach:
    # Use Strassen for the top-left 2x2 block (rows 0-1 of A, cols 0-1 of B, cols 0-1 of C)
    # and schoolbook for the rest.
    
    # Actually, let's just use schoolbook for now to ensure correctness, then optimize.
    
    # Schoolbook implementation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_sparse = [(u_idx(i, j), 1)]
                v_sparse = [(v_idx(j, k), 1)]
                w_sparse = [(w_idx(k, i), 1)]
                add_term(u_sparse, v_sparse, w_sparse)
    
    rank = len(terms_U)
    U = terms_U
    V = terms_V
    W = terms_W
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
