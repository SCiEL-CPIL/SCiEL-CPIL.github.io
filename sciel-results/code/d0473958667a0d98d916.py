"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll build U, V, W as lists of lists, then determine rank
    U = []  # Will be list of (n*m) x rank
    V = []  # Will be list of (m*p) x rank
    W = []  # Will be list of (p*n) x rank
    
    # Start with empty
    rank = 0
    U_rows = [[] for _ in range(n * m)]  # 8 rows
    V_rows = [[] for _ in range(m * p)]  # 16 rows
    W_rows = [[] for _ in range(p * n)]  # 8 rows
    
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal rank
        col = rank
        for idx in range(n * m):
            U_rows[idx].append(0)
        for idx in range(m * p):
            V_rows[idx].append(0)
        for idx in range(p * n):
            W_rows[idx].append(0)
        for idx, val in zip(u_indices, u_vals):
            U_rows[idx][col] = val
        for idx, val in zip(v_indices, v_vals):
            V_rows[idx][col] = val
        for idx, val in zip(w_indices, w_vals):
            W_rows[idx][col] = val
        rank += 1
    
    # Strassen helper for a 2x2x2 sub-block
    # a_idx[i][j] = global A index for local (i,j)
    # b_idx[i][j] = global B index for local (i,j)  
    # c_idx[i][j] = global C index for local (i,j)
    def strassen_block(a_idx, b_idx, c_idx):
        # a_idx[i][j] is the column in A for local row i, col j
        # In our A (2x4), global index is i*m + j = i*4 + j
        # b_idx[i][j] is global B index, B is 4x4, index is i*4 + j
        # c_idx[i][j] is global C index, C is 2x4, but W uses k*n+i = k*2+i
        # W[k*n+i, col] so for C[i,k], W index is k*n + i = k*2 + i
        
        # Local 2x2 matrices:
        # A_loc[i][j] = A[a_idx[i][j]]
        # B_loc[i][j] = B[b_idx[i][j]]
        # C_loc[i][j] = C[c_idx[i][j]]
        
        # Strassen:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # For each M, it's a product of (sum of A entries) * (sum of B entries)
        # We need to express each M as a rank-1 term in the tensor
        
        # M1: (A00+A11) * (B00+B11)
        # This expands to 4 terms, but in tensor decomposition, each "M" is actually
        # a bilinear form. Let me think...
        
        # Actually, in the tensor decomposition, each rank-1 term corresponds to
        # one product term. Strassen gives 7 bilinear forms, but each bilinear form
        # like (A00+A11)*(B00+B11) is itself a sum of 4 rank-1 terms.
        
        # Wait, no. In the standard tensor decomposition, we want:
        # C[i,k] = sum_r U[i*m+j... ] 
        # Actually the tensor is T[i,j,k] = 1 for C[i,k] += A[i,j]*B[j,k]
        # We decompose T = sum_r u_r ⊗ v_r ⊗ w_r where
        # (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i] * v_r[j] * w_r[k]
        
        # But our U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
        # So U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For a single rank-1 term, U[i*m+j, r] is nonzero for specific (i,j),
        # V[j*p+k, r] is nonzero for specific (j,k), W[k*n+i, r] is nonzero for specific (k,i).
        
        # Actually, each rank-1 term in the tensor decomposition of matrix mult:
        # The term contributes to C[i,k] the value U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # for some fixed (i,j,k). But that's not quite right either.
        
        # Let me reconsider. The tensor T has entries T[i,j,k] = delta (it's 1 for all valid i,j,k
        # in the sense that C[i,k] = sum_j A[i,j]B[j,k]).
        
        # A rank-1 decomposition: T = sum_r (u_r ⊗ v_r ⊗ w_r)
        # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i] * v_r[j] * w_r[k]
        
        # But u_r has n*m entries? No...
        
        # Looking at the interface: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # So U[a, r] where a = i*m+j, V[b, r] where b = j*p+k, W[c, r] where c = k*n+i
        
        # The reconstruction: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For this to equal sum_j A[i,j]*B[j,k], we need the sum over r and j to work out.
        
        # Each rank-1 term (column r) contributes:
        # sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For a "pure" rank-1 term that corresponds to a single (i0, j0, k0):
        # U[i0*m+j0, r] = 1, all other U[*, r] = 0
        # V[j0*p+k0, r] = 1, all other V[*, r] = 0
        # W[k0*n+i0, r] = 1, all other W[*, r] = 0
        # This gives contribution 1 to C[i0, k0] from A[i0,j0]*B[j0,k0].
        
        # For Strassen, each bilinear form like (A00+A11)*(B00+B11) corresponds to
        # multiple rank-1 terms. Specifically:
        # (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # That's 4 rank-1 terms!
        
        # So Strassen with 7 bilinear forms would give 7*4 = 28 rank-1 terms?
        # No wait, that's not right. Let me reconsider.
        
        # Actually, I think the way it works is:
        # Each "Strassen product" M = f(A) * g(B) where f and g are linear forms.
        # f(A) = sum of some A entries with coefficients
        # g(B) = sum of some B entries with coefficients
        # Then M = f(A)*g(B) is bilinear and can be written as a sum of rank-1 terms.
        
        # But in the tensor decomposition, we're decomposing the multiplication tensor,
        # not computing the products. Let me think about this differently.
        
        # The standard result: Strassen computes 2x2 matrix mult with 7 multiplications.
        # In the tensor decomposition sense, this means the tensor for 2x2x2 matrix mult
        # has rank at most 7.
        
        # For 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor T[i,j,k] = 1 (for all i,j,k in {0,1})
        # Rank-1 decomposition: T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
        
        # Where u_r[i], v_r[j], w_r[k] for i,j,k in {0,1}
        
        # The 7 Strassen terms:
        # M1: u=[1,1], v=[1,1], w=[1,0] -- no, this isn't quite right.
        
        # Let me look up the explicit rank-7 decomposition of the 2x2x2 matmul tensor.
        # 
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each M is a product of a linear form in A and a linear form in B.
        # In the tensor decomposition, each such product corresponds to a rank-1 term
        # IF the linear forms are single entries. But they're sums, so each M is actually
        # a sum of multiple rank-1 terms.
        
        # Hmm, I think I'm overcomplicating this. Let me just directly construct the
        # rank-1 terms for each Strassen bilinear form.
        
        # For M1 = (A00+A11)(B00+B11):
        # This equals A00*B00 + A00*B11 + A11*B00 + A11*B11
        # In terms of the tensor, this contributes to C via:
        # C00 += M1, C11 += M1
        # So the contribution to the tensor from M1 is:
        # For the A00*B00 part: contributes 1 to T[0,0,0] (C00) and 1 to T[0,0,1] (C11)
        # Wait, no. C00 = M1 + ..., C11 = M1 + ...
        # So M1 contributes +1 to both C00 and C11.
        # M1 = (A00+A11)(B00+B11)
        # The tensor contribution: for each (i,j) in A and (j',k) in B where j=j',
        # the coefficient in M1 is the product of the coefficient of A[i,j] in the A-part
        # and the coefficient of B[j,k] in the B-part.
        
        # A-part of M1: A00 + A11, so coeff[i,j] = 1 if (i,j) in {(0,0),(1,1)}, else 0
        # B-part of M1: B00 + B11, so coeff[j,k] = 1 if (j,k) in {(0,0),(1,1)}, else 0
        
        # The bilinear form M1 = sum_{i,j,k} A[i,j] * B[j,k] * coeff_A[i,j] * coeff_B[j,k]
        # But this is a sum of 4 rank-1 terms:
        # A00*B00: u[0,0]=1, v[0,0]=1, contributes to C00 and C11
        # A00*B11: u[0,0]=1, v[1,1]=1, contributes to C00 and C11
        # A11*B00: u[1,1]=1, v[0,0]=1, contributes to C00 and C11
        # A11*B11: u[1,1]=1, v[1,1]=1, contributes to C00 and C11
        
        # Wait, I think the issue is that each "Strassen multiplication" M is a scalar,
        # and it gets added to specific C entries. So in the tensor decomposition,
        # we need to track which C entries each M contributes to.
        
        # Let me just directly enumerate all the rank-1 terms.
        
        # For the full 2x4x4 problem, I'll directly construct the 28 terms.
        
        # Let me define the 4 blocks and for each, generate the 7 Strassen rank-1 terms.
        
        # Block 1: A[i,0:2], B[0:2,0:2], C[i,0:2] for i in {0,1}
        # Block 2: A[i,2:4], B[2:4,0:2], C[i,0:2] for i in {0,1}
        # Block 3: A[i,0:2], B[0:2,2:4], C[i,2:4] for i in {0,1}
        # Block 4: A[i,2:4], B[2:4,2:4], C[i,2:4] for i in {0,1}
        
        pass
    
    # I'll just hardcode the 28 terms directly.
    # Let me define them explicitly.
    
    # For each of the 4 blocks, and for each of the 7 Strassen terms,
    # I need to specify the U, V, W entries.
    
    # Let me create a list of 28 tuples, each specifying which U, V, W entries are nonzero.
    
    terms = []
    
    # Helper to add a block
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        # a_row_offset: which rows of A (0 or 1, but A has 2 rows, so offset is 0)
        # a_col_offset: which columns of A (0 or 2)
        # b_row_offset: which rows of B (0 or 2)
        # b_col_offset: which columns of B (0 or 2)
        # c_col_offset: which columns of C (0 or 2)
        
        # Local indices:
        # A local (i,j) -> global A[i, a_col_offset + j] -> U index i*4 + (a_col_offset + j)
        # B local (i,j) -> global B[b_row_offset + i, b_col_offset + j] -> V index (b_row_offset+i)*4 + (b_col_offset+j)
        # C local (i,j) -> global C[i, c_col_offset + j] -> W index (c_col_offset+j)*2 + i
        
        # Strassen 7 terms:
        # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
        # M2 = (A10+A11)B00, contributes to C10 and C11
        # M3 = A00(B01-B11), contributes to C01 and C11
        # M4 = A11(B10-B00), contributes to C00 and C10
        # M5 = (A00+A01)B11, contributes to C00 and C01
        # M6 = (A10-A00)(B00+B01), contributes to C11
        # M7 = (A01-A11)(B10+B11), contributes to C00
        
        # Each M is a bilinear form. To decompose into rank-1 terms,
        # I need to expand each product of linear forms.
        
        # Actually, I realize that in the tensor decomposition, each "multiplication"
        # in Strassen's algorithm corresponds to ONE rank-1 term in the tensor,
        # not multiple. The key insight is that the linear forms can have coefficients
        # other than 0 and 1, and the rank-1 term captures the entire bilinear form.
        
        # Let me reconsider. A rank-1 term in the tensor T[i,j,k] is:
        # u[i] * v[j] * w[k] for all i,j,k
        # In our parametrization:
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For a given rank-1 term r, U[*, r], V[*, r], W[*, r] are fixed vectors.
        # The contribution to C[i,k] is:
        # sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For this to represent a bilinear form like (A00+A11)(B00+B11) contributing
        # to specific C entries, we need:
        # sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 
        #   (sum of A entries with coeffs) * (sum of B entries with coeffs) * (C coefficient)
        
        # This is possible if we set:
        # U[i*m+j, r] = coeff_A[i,j] (for the A part)
        # V[j*p+k, r] = coeff_B[j,k] (for the B part)  
        # W[k*n+i, r] = coeff_C[i,k] (for the C part)
        
        # But this only works if the bilinear form is a SINGLE product of a linear form
        # in A, a linear form in B, and a linear form in C. And indeed, each Strassen M
        # is such a product, and it contributes to specific C entries.
        
        # Wait, but M1 contributes to BOTH C00 and C11. So the C part is a linear form
        # C00 + C11, meaning W has nonzero entries for both (k=0,i=0) and (k=1,i=1).
        
        # Let me verify: if U[i*m+j, r] = a_ij, V[j*p+k, r] = b_jk, W[k*n+i, r] = c_ik
        # Then contribution to C[i,k] = sum_j a_ij * b_jk * c_ik
        # For this to equal (sum_{i',j'} a_{i'j'} A[i',j']) * (sum_{j',k'} b_{j'k'} B[j',k']) * c_ik
        # We need c_ik to be the same for all i,k that M contributes to, and
        # the sum_j a_ij * b_jk to give the right cross terms.
        
        # Hmm, this is getting complicated. Let me just look up or derive the explicit
        # rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
        
        # The 2x2x2 matmul tensor T has T[i,j,k] = 1 for i,j,k in {0,1}.
        # A rank-7 decomposition: T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
        # where u_r, v_r, w_r are in R^2 (or R^4 for our parametrization).
        
        # In our parametrization, u_r is in R^{n*m} = R^4, v_r in R^{m*p} = R^4, w_r in R^{p*n} = R^4.
        # But for the 2x2x2 sub-problem, only certain entries are nonzero.
        
        # Let me just use the known Strassen decomposition. The 7 terms are:
        
        # I'll define each term by specifying which (i,j) pairs in A, (j,k) pairs in B,
        # and (k,i) pairs in C have nonzero values.
        
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # U: A00=1, A11=1 (indices 0, 3 in local 2x2)
        # V: B00=1, B11=1 (indices 0, 3 in local 2x2)
        # W: C00=1, C11=1 (indices 0, 3 in local 2x2, but W is k*n+i)
        
        # In global indices for block with offsets:
        # A local (0,0) -> global A[0, a_col_offset+0] -> U index 0*4 + (a_col_offset+0)
        # A local (0,1) -> global A[0, a_col_offset+1] -> U index 0*4 + (a_col_offset+1)
        # A local (1,0) -> global A[1, a_col_offset+0] -> U index 1*4 + (a_col_offset+0)
        # A local (1,1) -> global A[1, a_col_offset+1] -> U index 1*4 + (a_col_offset+1)
        
        # B local (0,0) -> global B[b_row_offset+0, b_col_offset+0] -> V index (b_row_offset+0)*4 + (b_col_offset+0)
        # etc.
        
        # C local (0,0) -> global C[0, c_col_offset+0] -> W index (c_col_offset+0)*2 + 0
        # C local (0,1) -> global C[0, c_col_offset+1] -> W index (c_col_offset+1)*2 + 0
        # C local (1,0) -> global C[1, c_col_offset+0] -> W index (c_col_offset+0)*2 + 1
        # C local (1,1) -> global C[1, c_col_offset+1] -> W index (c_col_offset+1)*2 + 1
        
        # Now, the 7 Strassen terms with their A, B, C coefficient patterns:
        
        # I'll represent each term as (a_coeffs, b_coeffs, c_coeffs) where
        # a_coeffs is a dict from local (i,j) to coefficient
        # b_coeffs is a dict from local (j,k) to coefficient
        # c_coeffs is a dict from local (i,k) to coefficient
        
        # M1: a={(0,0):1, (1,1):1}, b={(0,0):1, (1,1):1}, c={(0,0):1, (1,1):1}
        # M2: a={(1,0):1, (1,1):1}, b={(0,0):1}, c={(1,0):1, (1,1):-1}
        # M3: a={(0,0):1}, b={(0,1):1, (1,1):-1}, c={(0,1):1, (1,1):1}
        # M4: a={(1,1):1}, b={(1,0):1, (0,0):-1}, c={(0,0):1, (1,0):1}
        # M5: a={(0,0):1, (0,1):1}, b={(1,1):1}, c={(0,0):-1, (0,1):1}
        # M6: a={(1,0):1, (0,0):-1}, b={(0,0):1, (0,1):1}, c={(1,1):1}
        # M7: a={(0,1):1, (1,1):-1}, b={(1,0):1, (1,1):1}, c={(0,0):1}
        
        # Let me verify M1:
        # Contribution to C[i,k] = sum_j a[i,j] * b[j,k] * c[i,k]
        # C[0,0] = a[0,0]*b[0,0]*c[0,0] + a[0,0]*b[0,1]*c[0,0] + a[0,1]*b[1,0]*c[0,0] + a[0,1]*b[1,1]*c[0,0]
        # = 1*1*1 + 1*0*1 + 0*0*1 + 0*1*1 = 1 ✓
        # C[0,1] = a[0,0]*b[0,1]*c[0,1] + ... = 1*0*0 + ... = 0 ✓
        # C[1,0] = a[1,0]*b[0,0]*c[1,0] + ... = 0*1*0 + ... = 0 ✓
        # C[1,1] = a[1,0]*b[0,1]*c[1,1] + a[1,1]*b[1,0]*c[1,1] + a[1,1]*b[1,1]*c[1,1]
        # = 0*0*(-1) + 1*0*(-1) + 1*1*(-1) = -1 ✗
        
        # That's wrong! C[1,1] should be +1, not -1.
        
        # I think the issue is that the Strassen formulas give the VALUES of C,
        # but in the tensor decomposition, each rank-1 term contributes to ALL C entries
        # via the formula sum_j U[i*m+j] * V[j*p+k] * W[k*n+i].
        
        # Let me reconsider. The tensor T[i,j,k] for 2x2x2 matmul is 1 for all i,j,k.
        # We want T = sum_r u_r ⊗ v_r ⊗ w_r where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i] * v_r[j] * w_r[k].
        
        # But in our parametrization, it's different. We have:
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # And we want C[i,k] = sum_j A[i,j] * B[j,k].
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j] * B[j,k] (as a bilinear form)
        
        # For a single rank-1 term r, the contribution to C[i,k] from A[i,j]*B[j,k] is:
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # This should equal some bilinear form in A and B. For it to be a "pure" term,
        # we typically have U[i*m+j, r] nonzero for specific (i,j), V[j*p+k, r] nonzero for
        # specific (j,k), and W[k*n+i, r] nonzero for specific (k,i).
        
        # The key constraint is that for a fixed r, the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # as a function of (i,j,k) must be of the form f_r(i,j) * g_r(j,k) * h_r(k,i) for some
        # functions f, g, h. And we need sum_r f_r(i,j)*g_r(j,k)*h_r(k,i) = A[i,j]*B[j,k] (coefficient 1).
        
        # This is a very specific structure. Let me just look up the explicit rank-7 decomposition.
        
        # After some thought, I believe the correct approach is:
        # Each rank-1 term in the tensor decomposition of 2x2x2 matmul has the form:
        # u[i] * v[j] * w[k] where u, v, w are vectors in R^2 (for the 2x2x2 case).
        # In our parametrization:
        # U[i*m+j, r] = u_r[i] (same for all j? No...)
        
        # I think I need to be more careful. Let me just directly construct the 28 terms
        # by expanding each Strassen bilinear form into rank-1 terms.
        
        # For M1 = (A00+A11)(B00+B11) contributing to C00 and C11:
        # This is a bilinear form: (A00+A11)(B00+B11) * (delta_{C00} + delta_{C11})
        # = A00*B00*delta_{C00} + A00*B00*delta_{C11} + A00*B11*delta_{C00} + A00*B11*delta_{C11}
        #   + A11*B00*delta_{C00} + A11*B00*delta_{C11} + A11*B11*delta_{C00} + A11*B11*delta_{C11}
        
        # Each of these 8 terms is a rank-1 term! But that would give 8 terms per Strassen M,
        # and 7*8 = 56 total, which is way more than 28.
        
        # I think the correct interpretation is different. Let me reconsider.
        
        # Actually, I recall now: in the tensor decomposition, a rank-1 term u⊗v⊗w means
        # (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k]. For the 2x2x2 matmul tensor T[i,j,k]=1,
        # we need sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k.
        
        # In our parametrization, the "tensor" is not T[i,j,k] directly, but rather
        # the coefficients in C[i,k] = sum_j A[i,j]*B[j,k].
        
        # I think the cleanest way is to just verify numerically. Let me construct the
        # 28 terms and check that they correctly decompose the multiplication.
        
        # Let me try a different approach: just use the known fact that 2x2x2 matmul
        # has tensor rank 7, and construct the 7 terms for each of the 4 blocks.
        
        # I found a reference: the rank-7 decomposition of the 2x2x2 matmul tensor
        # can be written as follows. Let e0, e1 be standard basis.
        
        # Term 1: u=e0+e1, v=e0+e1, w=e0 (contributes to C00 only? No...)
        
        # I think I'm overcomplicating this. Let me just directly verify by construction.
        
        # For the 2x2x2 case, I'll define 7 rank-1 terms and verify they sum to the identity.
        
        # Let me use the following known decomposition (from Strassen):
        # The 7 terms are defined by their action on the 8 entries of T[i,j,k].
        
        # I'll just hardcode based on the Strassen formulas and verify.
        
        # Actually, let me just directly build the U, V, W matrices by expanding
        # each Strassen product into its constituent rank-1 terms.
        
        # For M1 = (A00+A11)(B00+B11):
        # This equals A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each product Aij*Bjk is a rank-1 term in the tensor.
        # A00*B00: U[0,0]=1, V[0,0]=1, and it contributes to C00 (since 0*4+0=0 for U, 0*4+0=0 for V, 0*2+0=0 for W)
        # But wait, A00*B00 contributes to C[0,0], not to C[1,1].
        
        # I think the issue is that in Strassen's algorithm, M1 is a scalar that gets
        # added to multiple C entries. In the tensor decomposition, we need to split
        # this into separate rank-1 terms for each C entry.
        
        # So M1 = (A00+A11)(B00+B11) contributes:
        # +1 to C00: this requires a rank-1 term where the product gives A00*B00 + A00*B11 + A11*B00 + A11*B11 all contributing to C00
        # +1 to C11: similarly for C11
        
        # But a single rank-1 term U[i*m+j]*V[j*p+k]*W[k*n+i] can only have a fixed pattern.
        
        # I think the resolution is that each "Strassen multiplication" actually corresponds
        # to MULTIPLE rank-1 terms in the tensor decomposition. Specifically, if M = f(A)*g(B)
        # where f is a sum of a_i A_i and g is a sum of b_j B_j, then M = sum_{i,j} a_i b_j A_i B_j,
        # which is a sum of |f|*|g| rank-1 terms.
        
        # For M1: |f|=2, |g|=2, so 4 rank-1 terms.
        # For M2: |f|=2, |g|=1, so 2 rank-1 terms.
        # For M3: |f|=1, |g|=2, so 2 rank-1 terms.
        # For M4: |f|=1, |g|=2, so 2 rank-1 terms.
        # For M5: |f|=2, |g|=1, so 2 rank-1 terms.
        # For M6: |f|=2, |g|=2, so 4 rank-1 terms.
        # For M7: |f|=2, |g|=2, so 4 rank-1 terms.
        
        # Total: 4+2+2+2+2+4+4 = 20 rank-1 terms per 2x2x2 block.
        # For 4 blocks: 80 terms. That's way more than 28.
        
        # I clearly have a misunderstanding. Let me reconsider.
        
        # I think the key insight is that in the tensor decomposition, a rank-1 term
        # u⊗v⊗w has (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k]. For the matmul tensor,
        # T[i,j,k] = 1 for all i,j,k (in the 2x2x2 case).
        
        # So we need sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k in {0,1}.
        
        # This is a different condition than what I was thinking. The tensor T has
        # 8 entries, all equal to 1. We need to decompose this 8-entry tensor into
        # a sum of rank-1 tensors.
        
        # A rank-1 tensor in this context is u⊗v⊗w where u, v, w are in R^2.
        # (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k].
        
        # So we need 7 vectors u_r, v_r, w_r in R^2 such that:
        # sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k in {0,1}.
        
        # This is a system of 8 equations (one for each (i,j,k)) with 7*3=21 unknowns.
        
        # From Strassen's algorithm, the 7 terms are:
        # (I'll use the standard Strassen decomposition)
        
        # Let me just look this up properly. The rank-7 decomposition of the 2x2x2
        # matrix multiplication tensor is well-known. One explicit decomposition is:
        
        # I'll define the 7 terms by their (u, v, w) vectors:
        
        # Actually, I realize that in our specific parametrization, the tensors are different.
        # Let me just directly construct and verify.
        
        # New approach: I'll construct the U, V, W matrices directly for the 2x4x4 problem
        # using 28 terms, and verify numerically that they work.
        
        # Let me define the 28 terms explicitly. I'll use the block structure.
        
        # For each of the 4 blocks, I'll define 7 terms.
        
        # Block parameters: (a_col_off, b_row_off, b_col_off, c_col_off)
        # Block 1: (0, 0, 0, 0)
        # Block 2: (2, 2, 0, 0)
        # Block 3: (0, 0, 2, 2)
        # Block 4: (2, 2, 2, 2)
        
        # For a block with these offsets, the 7 Strassen terms are:
        
        # I'll define each term as a tuple of (U_entries, V_entries, W_entries)
        # where each is a list of (index, value) pairs.
        
        # Let me define the 7 terms for a generic block:
        
        # Term 1 (M1): (A00+A11)(B00+B11) -> C00, C11
        # In the tensor sense, this is a sum of rank-1 terms. But I think for the
        # tensor decomposition, each "Strassen term" IS a rank-1 term in the tensor,
        # with the understanding that the linear forms have coefficients.
        
        # Let me just try: for M1, set:
        # U[i*m+j, r] = 1 if (i,j) in {(0,0), (1,1)} in the local block, else 0
        # V[j*p+k, r] = 1 if (j,k) in {(0,0), (1,1)} in the local block, else 0
        # W[k*n+i, r] = 1 if (k,i) in {(0,0), (1,1)} in the local block, else 0
        
        # Then the contribution to C[i,k] is:
        # sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For i=0, k=0: sum_j U[0*4+j, r] * V[j*4+0, r] * W[0*2+0, r]
        # = sum_j U[j, r] * V[j*4, r] * W[0, r]
        # U[j, r] is nonzero for j=0 (A[0,0]) and j=4+1=5? No, U index is i*4+j.
        # For i=0: U[0, r] = 1 (A[0,0]), U[1, r] = 0, U[2, r] = 0, U[3, r] = 0
        # Wait, U has n*m = 8 rows. U[i*4+j, r] for i in {0,1}, j in {0,1,2,3}.
        # For the block with a_col_off=0, the A entries are A[0,0], A[0,1], A[1,0], A[1,1].
        # U indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
        
        # I think I need to just carefully construct this. Let me write code that
        # builds the terms and then verifies.
        
        # Given the complexity, let me just directly construct the 28 terms
        # and verify numerically.
        
        pass
    
    # Let me take a completely different approach and just directly construct
    # the 28 rank-1 terms by expanding the Strassen algorithm.
    
    # I'll define a function that, given the block offsets, returns the 7 terms.
    # Each term is a dict with 'U', 'V', 'W' keys, each mapping global index to value.
    
    def get_strassen_terms(a_col_off, b_row_off, b_col_off, c_col_off):
        """Return 7 terms for a 2x2x2 block with given offsets."""
        terms = []
        
        # Local to global index mappings:
        # A local (i,j) -> U index: i*4 + (a_col_off + j)
        # B local (i,j) -> V index: (b_row_off + i)*4 + (b_col_off + j)
        # C local (i,k) -> W index: (c_col_off + k)*2 + i
        
        def u_idx(i, j):
            return i * 4 + (a_col_off + j)
        
        def v_idx(i, j):
            return (b_row_off + i) * 4 + (b_col_off + j)
        
        def w_idx(i, k):
            return (c_col_off + k) * 2 + i
        
        # Strassen 7 terms:
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # M2 = (A10+A11)B00 -> C10 += 1, C11 += -1
        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        # M5 = (A00+A01)B11 -> C00 += -1, C01 += 1
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        
        # For each term, I need to express it as a rank-1 term in the tensor.
        # A rank-1 term has U[*, r], V[*, r], W[*, r] such that
        # sum_j U[i*4+j, r] * V[j*4+k, r] * W[k*2+i, r] gives the right C[i,k].
        
        # For M1 = (A00+A11)(B00+B11) contributing to C00 and C11:
        # I'll set:
        # U[0, r] = 1 (A[0,0])
        # U[5, r] = 1 (A[1,1])  [i=1, j=1 -> 1*4+1=5]
        # V[0, r] = 1 (B[0,0])
        # V[3, r] = 1 (B[1,1])  [i=1, j=1 -> 1*4+1=5, but V index is (b_row_off+i)*4+(b_col_off+j)]
        # W[0, r] = 1 (C[0,0])  [k=0, i=0 -> 0*2+0=0]
        # W[3, r] = 1 (C[1,1])  [k=1, i=1 -> 1*2+1=3]
        
        # Let me verify: C[0,0] = sum_j U[0*4+j, r] * V[j*4+0, r] * W[0*2+0, r]
        # = U[0,r]*V[0,r]*W[0,r] + U[1,r]*V[4,r]*W[0,r] + U[2,r]*V[8,r]*W[0,r] + U[3,r]*V[12,r]*W[0,r]
        # = 1*1*1 + 0 + 0 + 0 = 1 ✓
        
        # C[0,1] = sum_j U[0*4+j, r] * V[j*4+1, r] * W[1*2+0, r]
        # = U[0,r]*V[1,r]*W[2,r] + U[1,r]*V[5,r]*W[2,r] + U[2,r]*V[9,r]*W[2,r] + U[3,r]*V[13,r]*W[2,r]
        # = 1*0*0 + 0 + 0 + 0 = 0 ✓
        
        # C[1,0] = sum_j U[1*4+j, r] * V[j*4+0, r] * W[0*2+1, r]
        # = U[4,r]*V[0,r]*W[1,r] + U[5,r]*V[4,r]*W[1,r] + U[6,r]*V[8,r]*W[1,r] + U[7,r]*V[12,r]*W[1,r]
        # = 0*1*0 + 1*0*0 + 0 + 0 = 0 ✓
        
        # C[1,1] = sum_j U[1*4+j, r] * V[j*4+1, r] * W[1*2+1, r]
        # = U[4,r]*V[1,r]*W[3,r] + U[5,r]*V[5,r]*W[3,r] + U[6,r]*V[9,r]*W[3,r] + U[7,r]*V[13,r]*W[3,r]
        # = 0*0*1 + 1*0*1 + 0 + 0 = 0 ✗
        
        # That's wrong! C[1,1] should be 1, not 0.
        
        # The issue is that V[5, r] should be nonzero for B[1,1], but V index for B[1,1]
        # is (b_row_off+1)*4 + (b_col_off+1). For block 1, b_row_off=0, b_col_off=0,
        # so V index = 1*4+1 = 5. And U[5, r] = 1 for A[1,1].
        
        # C[1,1] = sum_j U[4+j, r] * V[j*4+1, r] * W[3, r]
        # j=0: U[4,r]*V[1,r]*W[3,r] = 0*0*1 = 0
        # j=1: U[5,r]*V[5,r]*W[3,r] = 1*0*1 = 0
        # j=2: U[6,r]*V[9,r]*W[3,r] = 0
        # j=3: U[7,r]*V[13,r]*W[3,r] = 0
        # Total = 0 ✗
        
        # The problem is that V[j*4+1, r] for j=1 is V[5, r], which corresponds to
        # B[1,1] only if the V index formula is correct. V index = (b_row_off+i)*4 + (b_col_off+j).
        # For B[1,1]: i=1, j=1 -> (0+1)*4 + (0+1) = 5. ✓
        
        # But in the sum for C[1,1], we have V[j*4+1, r] where j is the middle index.
        # j=1 -> V[1*4+1, r] = V[5, r]. And V[5, r] should be 1 for B[1,1].
        
        # But I set V[3, r] = 1 for B[1,1]? No, I said V[3, r] = 1 (B[1,1]) but that's wrong.
        # V index for B[1,1] is 5, not 3.
        
        # Let me redo: V[5, r] = 1 for B[1,1].
        
        # C[1,1] = sum_j U[4+j, r] * V[j*4+1, r] * W[3, r]
        # j=1: U[5,r]*V[5,r]*W[3,r] = 1*1*1 = 1 ✓
        
        # Great! So I had the V index wrong. Let me redo all terms carefully.
        
        # Term 1 (M1):
        # U: A[0,0]=1 -> U[0, r]=1; A[1,1]=1 -> U[5, r]=1
        # V: B[0,0]=1 -> V[0, r]=1; B[1,1]=1 -> V[5, r]=1
        # W: C[0,0]=1 -> W[0, r]=1; C[1,1]=1 -> W[3, r]=1
        
        # Let me verify all C entries:
        # C[0,0] = U[0]*V[0]*W[0] = 1*1*1 = 1 ✓
        # C[0,1] = U[0]*V[1]*W[2] + U[1]*V[5]*W[2] + ... = 1*0*0 + 0*1*0 + ... = 0 ✓
        # C[1,0] = U[4]*V[0]*W[1] + U[5]*V[4]*W[1] + ... = 0*1*0 + 1*0*0 + ... = 0 ✓
        # C[1,1] = U[4]*V[1]*W[3] + U[5]*V[5]*W[3] + ... = 0*0*1 + 1*1*1 + ... = 1 ✓
        
        # Now for the other blocks, the indices will be different due to offsets.
        
        # Let me define all 7 terms for a block:
        
        # Term 1 (M1): (A00+A11)(B00+B11) -> C00, C11
        terms.append({
            'U': {u_idx(0,0): 1, u_idx(1,1): 1},
            'V': {v_idx(0,0): 1, v_idx(1,1): 1},
            'W': {w_idx(0,0): 1, w_idx(1,1): 1}
        })
        
        # Term 2 (M2): (A10+A11)B00 -> C10, C11
        # C10 += 1, C11 += -1
        terms.append({
            'U': {u_idx(1,0): 1, u_idx(1,1): 1},
            'V': {v_idx(0,0): 1},
            'W': {w_idx(1,0): 1, w_idx(1,1): -1}
        })
        
        # Term 3 (M3): A00(B01-B11) -> C01, C11
        # C01 += 1, C11 += 1
        terms.append({
            'U': {u_idx(0,0): 1},
            'V': {v_idx(0,1): 1, v_idx(1,1): -1},
            'W': {w_idx(0,1): 1, w_idx(1,1): 1}
        })
        
        # Term 4 (M4): A11(B10-B00) -> C00, C10
        # C00 += 1, C10 += 1
        terms.append({
            'U': {u_idx(1,1): 1},
            'V': {v_idx(1,0): 1, v_idx(0,0): -1},
            'W': {w_idx(0,0): 1, w_idx(1,0): 1}
        })
        
        # Term 5 (M5): (A00+A01)B11 -> C00, C01
        # C00 += -1, C01 += 1
        terms.append({
            'U': {u_idx(0,0): 1, u_idx(0,1): 1},
            'V': {v_idx(1,1): 1},
            'W': {w_idx(0,0): -1, w_idx(0,1): 1}
        })
        
        # Term 6 (M6): (A10-A00)(B00+B01) -> C11
        # C11 += 1
        terms.append({
            'U': {u_idx(1,0): 1, u_idx(0,0): -1},
            'V': {v_idx(0,0): 1, v_idx(0,1): 1},
            'W': {w_idx(1,1): 1}
        })
        
        # Term 7 (M7): (A01-A11)(B10+B11) -> C00
        # C00 += 1
        terms.append({
            'U': {u_idx(0,1): 1, u_idx(1,1): -1},
            'V': {v_idx(1,0): 1, v_idx(1,1): 1},
            'W': {w_idx(0,0): 1}
        })
        
        return terms
    
    # Now build the full U, V, W for all 4 blocks
    blocks = [
        (0, 0, 0, 0),  # Block 1
        (2, 2, 0, 0),  # Block 2
        (0, 0, 2, 2),  # Block 3
        (2, 2, 2, 2),  # Block 4
    ]
    
    all_terms = []
    for block in blocks:
        all_terms.extend(get_strassen_terms(*block))
    
    rank = len(all_terms)
    
    # Build U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(all_terms):
        for idx, val in term['U'].items():
            U[idx][r] = val
        for idx, val in term['V'].items():
            V[idx][r] = val
        for idx, val in term['W'].items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
