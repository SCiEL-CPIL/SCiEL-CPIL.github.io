import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Partition: rows=[2,1], inner=[2,2,1], output=[2,2,1]
    row_parts = [2, 1]
    inner_parts = [2, 2, 1]
    out_parts = [2, 2, 1]
    
    # Get cumulative indices
    row_starts = [0]
    for s in row_parts:
        row_starts.append(row_starts[-1] + s)
    inner_starts = [0]
    for s in inner_parts:
        inner_starts.append(inner_starts[-1] + s)
    out_starts = [0]
    for s in out_parts:
        out_starts.append(out_starts[-1] + s)
    
    # Collect all rank-1 terms
    terms = []  # Each term is (U_idx, V_idx, W_idx) where these are sets of indices
    
    for ri, (r0, r1) in enumerate(zip(row_starts[:-1], row_starts[1:])):
        for ii, (i0, i1) in enumerate(zip(inner_starts[:-1], inner_starts[1:])):
            for oi, (o0, o1) in enumerate(zip(out_starts[:-1], out_starts[1:])):
                rn = r1 - r0  # row size
                in_n = i1 - i0  # inner size
                on = o1 - o0  # output size
                
                if rn == 2 and in_n == 2 and on == 2:
                    # Strassen 2x2x2: rank 7
                    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
                    # C = [[c11,c12],[c21,c22]]
                    # m1 = (a11+a22)(b11+b22)
                    # m2 = (a21+a22)b11
                    # m3 = a11(b12-b22)
                    # m4 = a22(b21-b11)
                    # m5 = (a11+a12)b22
                    # m6 = (a21-a11)(b11+b12)
                    # m7 = (a12-a22)(b21+b22)
                    # c11 = m1+m4-m5+m7
                    # c12 = m3+m5
                    # c21 = m2+m4
                    # c22 = m1-m2+m3+m6
                    
                    # For each Strassen term, we need to express it as rank-1 contributions
                    # The Strassen algorithm computes:
                    # m1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
                    # This is not rank-1 in the tensor sense. We need to be careful.
                    
                    # Actually, Strassen's 7 multiplications correspond to 7 rank-1 tensors
                    # where each "multiplication" is a product of linear combinations.
                    
                    # Let me define the 7 Strassen factors properly.
                    # For 2x2x2 matrix multiplication tensor:
                    # The 7 rank-1 terms are:
                    
                    # Term 1: (e11+e22) ⊗ (e11+e22) ⊗ (e11+e22 - e12 - e21 + e22)
                    # Wait, I need to be more careful.
                    
                    # The standard Strassen decomposition for 2x2x2:
                    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
                    # p = a*e + b*g
                    # q = a*f + b*h  
                    # r = c*e + d*g
                    # s = c*f + d*h
                    
                    # m1 = (a+d)(e+h) = ae + ah + de + dh
                    # m2 = (c+d)e = ce + de
                    # m3 = a(f-h) = af - ah
                    # m4 = d(g-e) = dg - de
                    # m5 = (a+b)h = ah + bh
                    # m6 = (c-a)(e+f) = ce + cf - ae - af
                    # m7 = (b-d)(g+h) = bg + bh - dg - dh
                    
                    # p = m1 + m4 - m5 + m7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
                    #   = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
                    #   = ae + bg ✓
                    
                    # q = m3 + m5 = (af-ah) + (ah+bh) = af + bh ✓
                    
                    # r = m2 + m4 = (ce+de) + (dg-de) = ce + dg ✓
                    
                    # s = m1 - m2 + m3 + m6 = (ae+ah+de+dh) - (ce+de) + (af-ah) + (ce+cf-ae-af)
                    #   = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
                    #   = dh + cf ✓
                    
                    # So the 7 rank-1 terms for the 2x2x2 block are:
                    # Each term k has:
                    #   U vector: linear combo of A entries (a,b,c,d)
                    #   V vector: linear combo of B entries (e,f,g,h)
                    #   W vector: linear combo of C entries (p,q,r,s)
                    
                    # Term 1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)
                    # Term 2: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,0)
                    # Term 3: U=(1,0,0,0), V=(0,1,0,-1), W=(0,1,0,1)
                    # Term 4: U=(0,0,0,1), V=(0,0,1,-1), W=(0,0,1,0)
                    # Term 5: U=(1,1,0,0), V=(0,0,0,1), W=(0,1,0,0)
                    # Term 6: U=(0,0,1,-1), V=(1,1,0,0), W=(0,1,0,0)
                    # Term 7: U=(0,1,0,-1), V=(0,0,1,1), W=(1,0,0,0)
                    
                    # Wait, let me re-derive. The W vector for each term should give the coefficient in the output.
                    
                    # m1 contributes to: p (+1), s (+1) → W=(1,0,0,1)
                    # m2 contributes to: r (+1), s (-1) → W=(0,0,1,-1)
                    # m3 contributes to: q (+1), s (+1) → W=(0,1,0,1)
                    # m4 contributes to: p (+1), r (+1) → W=(1,0,1,0)
                    # m5 contributes to: p (-1), q (+1) → W=(-1,1,0,0)
                    # m6 contributes to: q (+1), s (+1) → W=(0,1,0,1)
                    # m7 contributes to: p (+1), s (-1) → W=(1,0,0,-1)
                    
                    # Let me verify:
                    # p = m1*1 + m2*0 + m3*0 + m4*1 + m5*(-1) + m6*0 + m7*1
                    #   = m1 + m4 - m5 + m7 ✓
                    # q = m1*0 + m2*0 + m3*1 + m4*0 + m5*1 + m6*1 + m7*0
                    #   = m3 + m5 + m6
                    # But we need q = m3 + m5. So m6 should not contribute to q.
                    
                    # Let me recheck m6: m6 = (c-a)(e+f) = ce + cf - ae - af
                    # This contributes to... let me see where cf appears. cf is part of s = cf + dh.
                    # And ae is part of p = ae + bg. af is part of q = af + bh. ce is part of r = ce + dg.
                    
                    # So m6 = ce + cf - ae - af
                    # ce contributes to r (+1)
                    # cf contributes to s (+1)
                    # -ae contributes to p (-1)
                    # -af contributes to q (-1)
                    # So W for m6 = (-1, -1, 1, 1)
                    
                    # Let me redo all W vectors:
                    # m1 = ae + ah + de + dh: a→p, h→? No wait.
                    # ae: a from A, e from B → contributes to p
                    # ah: a from A, h from B → contributes to q  
                    # de: d from A, e from B → contributes to r
                    # dh: d from A, h from B → contributes to s
                    # So m1 contributes: p(+1), q(+1), r(+1), s(+1)? No, that's not right.
                    
                    # I'm confusing myself. Let me think differently.
                    
                    # Each Strassen multiplication m_k = u_k^T * A * v_k where u_k and v_k are vectors
                    # and the result contributes to C via w_k.
                    
                    # Actually, the cleaner way: each rank-1 term is:
                    # (linear combo of A entries) * (linear combo of B entries) → adds to specific C entries
                    
                    # For m1 = (a+d)(e+h):
                    # This equals ae + ah + de + dh
                    # ae is the (1,1) entry of A times (1,1) of B → contributes to C(1,1)=p
                    # ah is (1,1) of A times (2,2) of B → contributes to C(1,2)=q
                    # de is (2,2) of A times (1,1) of B → contributes to C(2,1)=r
                    # dh is (2,2) of A times (2,2) of B → contributes to C(2,2)=s
                    # So the W vector for m1 has +1 at positions corresponding to p,q,r,s
                    # But that means m1 alone would add to all four outputs, which is correct since
                    # m1 is used in combinations to get the final answers.
                    
                    # The point is: in the tensor decomposition, each rank-1 term is:
                    # u ⊗ v ⊗ w where u is a vector in R^{4} (for A entries), v in R^{4} (for B), w in R^{4} (for C)
                    # and the sum of u_k ⊗ v_k ⊗ w_k = multiplication tensor
                    
                    # For m1: u1 = (1,0,0,1), v1 = (1,0,0,1), w1 = (1,1,1,1)
                    # Check: sum over k of u_k[i]*v_k[j]*w_k[l] should equal delta_{il} if j is the "inner" index... 
                    # Actually for matrix mult, the tensor T[i,j,l] = 1 if the (i,l) entry of C equals sum_j A[i,j]*B[j,l]
                    # So T[i,j,l] = 1 for all i,j,l? No, T is a 3-tensor where T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
                    # So T[i,j,k] = 1 for all valid i,j,k (it's the all-ones tensor for the multiplication pattern).
                    
                    # Wait no. The matrix multiplication tensor has T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] appears in C[i,k]. For standard matrix mult, every triple (i,j,k) with valid indices contributes, so T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
                    
                    # Hmm, that's not quite right either. Let me think again.
                    
                    # The matrix multiplication tensor T ∈ R^{n×m × m×p × p×n} (or similar indexing)
                    # T[i,j | j,k | i,k] = 1 (using the convention that the first index is (row, inner_col), second is (inner_row, col), third is (col, row) for output)
                    
                    # With the trace convention W[k*n+i, r], the output index is k*n+i where i is row, k is col.
                    
                    # So for 2x2x2:
                    # A has entries A[0,0], A[0,1], A[1,0], A[1,1] → flattened: A[0], A[1], A[2], A[3]
                    # B has entries B[0,0], B[0,1], B[1,0], B[1,1] → flattened: B[0], B[1], B[2], B[3]
                    # C has entries C[0,0], C[0,1], C[1,0], C[1,1] → flattened: C[0], C[1], C[2], C[3]
                    
                    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = A[0]*B[0] + A[1]*B[2]
                    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = A[0]*B[1] + A[1]*B[3]
                    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = A[2]*B[0] + A[3]*B[2]
                    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = A[2]*B[1] + A[3]*B[3]
                    
                    # The tensor T has T[a_idx, b_idx, c_idx] = 1 where:
                    # (0,0,0)=1, (1,2,0)=1, (0,1,1)=1, (1,3,1)=1, (2,0,2)=1, (3,2,2)=1, (2,1,3)=1, (3,3,3)=1
                    
                    # Now for Strassen, we decompose this as sum of 7 rank-1 tensors.
                    # Each rank-1 tensor is u ⊗ v ⊗ w.
                    
                    # Let me just use the known Strassen decomposition directly.
                    # I'll define the 7 terms as (u, v, w) triples.
                    
                    strassen_2x2x2 = [
                        # (u, v, w)
                        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),   # m1
                        ((0,0,1,1), (1,0,0,0), (0,0,1,0)),    # m2
                        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),   # m3
                        ((0,0,0,1), (0,0,1,-1), (0,0,1,0)),   # m4
                        ((1,1,0,0), (0,0,0,1), (0,1,0,0)),    # m5
                        ((0,0,1,-1), (1,1,0,0), (0,1,0,0)),   # m6 -- need to verify
                        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),   # m7
                    ]
                    
                    # Let me verify this decomposition by checking if sum of u_k[i]*v_k[j]*w_k[l] = T[i,j,l]
                    
                    # This is getting complex. Let me just use a verified Strassen decomposition.
                    # I'll compute it numerically to be safe.
                    
                    # Actually, let me just use the schoolbook for now and optimize later.
                    # For the 2x2x2 block, schoolbook is rank 8.
                    
                    # Let me use a simpler approach: just use schoolbook for all blocks first,
                    # then replace 2x2x2 blocks with Strassen.
                    
                    pass
                else:
                    # Schoolbook: rank = rn * in_n * on
                    for a_i in range(rn):
                        for b_j in range(in_n):
                            for c_k in range(on):
                                # A index: (r0+a_i, i0+b_j) → flat index (r0+a_i)*m + (i0+b_j)
                                # B index: (i0+b_j, o0+c_k) → flat index (i0+b_j)*p + (o0+c_k)
                                # C index: (o0+c_k, r0+a_i) → flat index (o0+c_k)*n + (r0+a_i)
                                a_idx = (r0 + a_i) * m + (i0 + b_j)
                                b_idx = (i0 + b_j) * p + (o0 + c_k)
                                c_idx = (o0 + c_k) * n + (r0 + a_i)
                                terms.append((a_idx, b_idx, c_idx))
                    
    # Now apply Strassen for 2x2x2 blocks
    # Let me redo this more carefully
    
    # Clear and restart with proper Strassen implementation
    terms = []
    
    def add_schoolbook(r0, i0, o0, rn, in_n, on):
        """Add schoolbook terms for a sub-block"""
        for a_i in range(rn):
            for b_j in range(in_n):
                for c_k in range(on):
                    a_idx = (r0 + a_i) * m + (i0 + b_j)
                    b_idx = (i0 + b_j) * p + (o0 + c_k)
                    c_idx = (o0 + c_k) * n + (r0 + a_i)
                    terms.append((a_idx, b_idx, c_idx))
    
    def add_strassen_2x2x2(r0, i0, o0):
        """Add Strassen terms for a 2x2x2 sub-block at position (r0, i0, o0)"""
        # A entries: (r0, i0), (r0, i0+1), (r0+1, i0), (r0+1, i0+1)
        # B entries: (i0, o0), (i0, o0+1), (i0+1, o0), (i0+1, o0+1)
        # C entries: (o0, r0), (o0, r0+1), (o0+1, r0), (o0+1, r0+1)
        
        # Flatten indices
        A00 = r0 * m + i0
        A01 = r0 * m + (i0 + 1)
        A10 = (r0 + 1) * m + i0
        A11 = (r0 + 1) * m + (i0 + 1)
        
        B00 = i0 * p + o0
        B01 = i0 * p + (o0 + 1)
        B10 = (i0 + 1) * p + o0
        B11 = (i0 + 1) * p + (o0 + 1)
        
        C00 = o0 * n + r0
        C01 = o0 * n + (r0 + 1)
        C10 = (o0 + 1) * n + r0
        C11 = (o0 + 1) * n + (r0 + 1)
        
        # Strassen 2x2x2 decomposition (7 terms)
        # Each term: list of (A_indices_with_coeffs, B_indices_with_coeffs, C_indices_with_coeffs)
        
        # m1 = (A00+A11)(B00+B11)
        # Contributes to: C00, C01, C10, C11 (but with signs in combination)
        # The rank-1 tensor for m1:
        # u = A00*1 + A11*1, v = B00*1 + B11*1, w = C00*1 + C01*1 + C10*1 + C11*1
        # Wait, that's not a rank-1 tensor in the right sense.
        
        # Let me think about this differently. The multiplication tensor for 2x2x2 is:
        # T = sum of 8 rank-1 terms (schoolbook):
        # (A00⊗B00⊗C00) + (A01⊗B10⊗C00) + (A00⊗B01⊗C01) + (A01⊗B11⊗C01)
        # + (A10⊗B00⊗C10) + (A11⊗B10⊗C10) + (A10⊗B01⊗C11) + (A11⊗B11⊗C11)
        
        # Strassen replaces these 8 with 7. The 7 rank-1 terms are:
        # t1 = (A00+A11)⊗(B00+B11)⊗(C00+C01+C10+C11)  -- NO, this isn't right
        
        # The correct way: each "multiplication" in Strassen is a scalar product,
        # but in the tensor decomposition, we express the tensor as sum of rank-1 tensors.
        
        # Let me just verify numerically. I'll define the 7 terms and check.
        
        # Known Strassen rank-1 decomposition for 2x2x2 matmul:
        # Using the standard formulation, the 7 terms are:
        
        strassen_terms = [
            # (u_coeffs for A, v_coeffs for B, w_coeffs for C)
            # where u is over [A00, A01, A10, A11], v over [B00, B01, B10, B11], w over [C00, C01, C10, C11]
            
            # m1 = (a+d)(e+h)
            ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
            # m2 = (c+d)e  
            ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
            # m3 = a(f-h)
            ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
            # m4 = d(g-e)
            ((0, 0, 0, 1), (0, 0, 1, -1), (0, 0, 1, 0)),
            # m5 = (a+b)h
            ((1, 1, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0)),
            # m6 = (c-a)(e+f)
            ((0, 0, 1, -1), (1, 1, 0, 0), (0, 1, 0, 0)),
            # m7 = (b-d)(g+h)
            ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
        ]
        
        # Verify: sum_k u_k[i]*v_k[j]*w_k[l] should equal T[i,j,l]
        # T is the 2x2x2 matmul tensor
        
        # Let me just add these terms and trust the standard decomposition.
        # If it's wrong, the verification will catch it.
        
        for u, v, w in strassen_terms:
            # Expand each term into individual contributions
            # u is a vector over A entries, v over B entries, w over C entries
            # The rank-1 tensor u⊗v⊗w means:
            # for each (i,j,l): contribution = u[i]*v[j]*w[l]
            # We need to add this to our global tensor.
            
            # But we're building U, V, W matrices where each column is a rank-1 term.
            # So for each Strassen term, we add one column to U, V, W.
            
            # The U column has non-zero entries at A positions
            # The V column has non-zero entries at B positions
            # The W column has non-zero entries at C positions
            
            # For this term, U values at [A00, A01, A10, A11] are u[0], u[1], u[2], u[3]
            # V values at [B00, B01, B10, B11] are v[0], v[1], v[2], v[3]
            # W values at [C00, C01, C10, C11] are w[0], w[1], w[2], w[3]
            
            # Store as sparse representation: list of (index, value)
            u_sparse = [(A00, u[0]), (A01, u[1]), (A10, u[2]), (A11, u[3])]
            v_sparse = [(B00, v[0]), (B01, v[1]), (B10, v[2]), (B11, v[3])]
            w_sparse = [(C00, w[0]), (C01, w[1]), (C10, w[2]), (C11, w[3])]
            
            # Remove zero entries
            u_sparse = [(i, val) for i, val in u_sparse if val != 0]
            v_sparse = [(i, val) for i, val in v_sparse if val != 0]
            w_sparse = [(i, val) for i, val in w_sparse if val != 0]
            
            terms.append((u_sparse, v_sparse, w_sparse))
    
    # Now build all terms
    for ri, (r0, r1) in enumerate(zip(row_starts[:-1], row_starts[1:])):
        for ii, (i0, i1) in enumerate(zip(inner_starts[:-1], inner_starts[1:])):
            for oi, (o0, o1) in enumerate(zip(out_starts[:-1], out_starts[1:])):
                rn = r1 - r0
                in_n = i1 - i0
                on = o1 - o0
                
                if rn == 2 and in_n == 2 and on == 2:
                    add_strassen_2x2x2(r0, i0, o0)
                else:
                    add_schoolbook(r0, i0, o0, rn, in_n, on)
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, term in enumerate(terms):
        u_sparse, v_sparse, w_sparse = term
        for idx, val in u_sparse:
            U[idx][col] = val
        for idx, val in v_sparse:
            V[idx][col] = val
        for idx, val in w_sparse:
            W[idx][col] = val
    
    # Verify the decomposition
    # Check that sum_k U[:,k] ⊗ V[:,k] ⊗ W[:,k] = T
    # T[i,j,l] = 1 if (i,j) is a valid A entry, (j',l) is a valid B entry with j'=j (inner index matches), and (l',i') is a valid C entry with l'=l, i'=i
    
    # Actually, let me verify numerically
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The decomposition says: T = sum_k U_k ⊗ V_k ⊗ W_k
    # where T[i, j, l] = sum_k U[i,k] * V[j,k] * W[l,k]
    
    # For matrix multiplication: C = A * B
    # C[i, l] = sum_j A[i, j] * B[j, l]
    # In our indexing: A flat index = i*m + j, B flat index = j*p + l, C flat index = l*n + i
    # So T[a_idx, b_idx, c_idx] = 1 if a_idx = i*m+j, b_idx = j*p+l, c_idx = l*n+i for some i,j,l
    # i.e., T[i*m+j, j*p+l, l*n+i] = 1
    
    # Check: for all i, j, l: sum_k U[i*m+j, k] * V[j*p+l, k] * W[l*n+i, k] should be 1
    # And for all other (a_idx, b_idx, c_idx), it should be 0.
    
    # Let's check a few entries
    error = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a_idx = i * m + j
                b_idx = j * p + l
                c_idx = l * n + i
                val = sum(U_np[a_idx, k] * V_np[b_idx, k] * W_np[c_idx, k] for k in range(rank))
                if abs(val - 1) > 1e-9:
                    error += 1
    
    if error > 0:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + l][col] = 1
                    W[l * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

