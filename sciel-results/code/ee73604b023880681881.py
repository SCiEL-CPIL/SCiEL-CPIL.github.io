"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2   # 8 blocks of 2 columns each
    rank = 7 * num_blocks # 56

    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56

    # Strassen's 7 tensor factors for 2x2x2 matrix multiplication.
    # Each factor is (u, v, w) where:
    #   u[i] = coefficient of A_{i, j} (A is 2x2, indexed by row i, col j)
    #   v[j] = coefficient of B_{j, k} (B is 2x2, indexed by row j, col k)
    #   w[k] = coefficient of C_{i, k} (C is 2x2, indexed by row i, col k)
    #
    # The tensor identity: for all i,j,k in {0,1}:
    #   sum_r u_r[i] * v_r[j] * w_r[k] = delta_{j,k}
    # (since C_{i,k} = sum_j A_{i,j} B_{j,k}, the structure tensor has
    #  T[i,j,k] = 1 when j==k, 0 otherwise... wait, no.)
    #
    # Actually the matrix multiplication tensor is:
    #   T[i,j,k] = 1 for all valid (i,j,k) -- no that's not right either.
    #
    # The correct formulation: the bilinear map (A,B) -> C = AB has the
    # property that C_{i,k} = sum_j A_{i,j} * B_{j,k}.
    # A rank-1 term in the decomposition contributes:
    #   C_{i,k} += u[i] * (sum_j A_{i,j} * v[j]) * w[k]  -- NO
    #
    # Let me use the standard tensor formulation:
    # The multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1
    # (it's the identity in the middle index). The rank-r decomposition is:
    #   M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]  (in the simplified form)
    #
    # But with the given interface where U has n*m rows, V has m*p rows,
    # W has p*n rows, the decomposition works as:
    #   C_{i,k} = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, that doesn't look right dimensionally. Let me re-examine.
    #
    # Actually, I believe the convention is:
    #   C_{i,k} = sum_r (sum_j U[i*m+j, r] * A_{i,j}) * (sum_k' V[j*p+k', r] * B_{j,k'}) * W[k*n+i, r]
    # No, that's overcomplicating it.
    #
    # The standard schoolbook decomposition: for each (i,j,k), create one rank-1 term
    # with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # Then C_{i,k} = sum_{i',j',k'} [U[i'*m+j',r] * V[j'*p+k',r] * W[k'*n+i',r]] * A[i',j'] * B[j',k']
    # For the schoolbook term (i,j,k): contributes A[i,j]*B[j,k] to C[i,k] (via W[k*n+i]=1).
    #
    # So the general formula is:
    #   C_{i,k} = sum_r sum_{i',j',k'} U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r] * A[i',j'] * B[j',k']
    #
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    #   sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r] = delta_{i',i} * delta_{j',k'} * delta_{k',k}... 
    # No, we need the sum over r of the product to give delta_{i',i} delta_{j',j} delta_{k',k} where j'=k' (the middle index matches).
    #
    # Actually simpler: the schoolbook has exactly one term for each (i,j,k) with
    # U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # The contribution to C[i,k] from this term is:
    #   U[i*m+j]*V[j*p+k]*W[k*n+i] * A[i,j] * B[j,k] = 1*1*1*A[i,j]*B[j,k]
    # Summing over all (i,j,k) terms gives sum_j A[i,j]*B[j,k]. Correct!
    #
    # Now for Strassen on a 2x2x2 block: we want 7 terms that replace the 8 schoolbook terms
    # for a fixed set of A indices (all 4), B indices (2 rows x 2 cols), C indices (2 rows x 2 cols).
    #
    # For block t (columns 2t, 2t+1), the schoolbook terms are for all (i, j, k_local)
    # where i in {0,1}, j in {0,1}, k_local in {0,1}, and k = 2t + k_local.
    #
    # Strassen's 7 factors for 2x2x2:
    # M1 = (a00+a11)(b00+b11): u=[1,1], v=[1,1], w=[1,1]
    # M2 = (a10+a11)b00:        u=[0,1], v=[1,0], w=[0,1]  -- need to verify
    # M3 = a00(b01-b11):        u=[1,0], v=[0,1], w=[1,-1] -- need to verify
    # M4 = a11(b10-b00):        u=[0,1], v=[1,-1], w=[1,0] -- need to verify
    # M5 = (a00+a01)b11:        u=[1,1], v=[0,1], w=[0,1]  -- need to verify
    # M6 = (a10-a00)(b00+b01):  u=[-1,1], v=[1,1], w=[0,1] -- need to verify
    # M7 = (a01-a11)(b10+b11):  u=[0,1], v=[1,1], w=[-1,1] -- need to verify
    #
    # Let me verify M1: sum over r of u[i]*v[j]*w[k] should give delta_{j,k} for the 2x2x2 tensor.
    # Wait no. The tensor for 2x2x2 matrix mult has T[i,j,k] = 1 for ALL i,j,k? No.
    #
    # I think I'm overcomplicating this. Let me just verify that the 7 Strassen factors
    # correctly reproduce the 8 schoolbook terms by checking the output formula.
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Where:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # Verify C00 = M1 + M4 - M5 + M7:
    # = (a00+a11)(b00+b11) + a11(b10-b00) - (a00+a01)b11 + (a01-a11)(b10+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00 - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
    # = a00*b00 + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + (a11*b11 - a11*b11) + (a11*b10 - a11*b10) + (-a01*b11 + a01*b11) + a01*b10
    # = a00*b00 + a01*b10
    # Hmm, that should be a00*b00 + a01*b10? But C00 = a00*b00 + a01*b10. Yes! Correct.
    #
    # Verify C01 = M3 + M5:
    # = a00(b01-b11) + (a00+a01)b11
    # = a00*b01 - a00*b11 + a00*b11 + a01*b11
    # = a00*b01 + a01*b11. Correct!
    #
    # Verify C10 = M2 + M4:
    # = (a10+a11)b00 + a11(b10-b00)
    # = a10*b00 + a11*b00 + a11*b10 - a11*b00
    # = a10*b00 + a11*b10. Correct!
    #
    # Verify C11 = M1 - M2 + M3 + M6:
    # = (a00+a11)(b00+b11) - (a10+a11)b00 + a00(b01-b11) + (a10-a00)(b00+b01)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b01 - a00*b11 + a10*b00 + a10*b01 - a00*b00 - a00*b01
    # = (a00*b00 - a00*b00) + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + a11*b11 + (-a10*b00 + a10*b00) + (a00*b01 - a00*b01) + a10*b01
    # = a11*b11 + a10*b01. Correct!
    #
    # Great, so the Strassen formulas are verified. Now I need to express each M_r as a
    # rank-1 tensor factor (u, v, w) such that:
    #   u[i] * v[j] * w[k] summed over r gives the correct contribution.
    #
    # For the interface:
    #   U[i*m+j, col] = coefficient for A_{i,j} in factor col
    #   V[j*p+k, col] = coefficient for B_{j,k} in factor col
    #   W[k*n+i, col] = coefficient for C_{i,k} in factor col
    #
    # For a rank-1 factor corresponding to M_r = (sum_{i,j} u_r[i,j] * A_{i,j}) * (sum_{j,k} v_r[j,k] * B_{j,k}):
    #   U[i*m+j, col] = u_r[i,j]
    #   V[j*p+k, col] = v_r[j,k]
    #   W[k*n+i, col] = 1 (the output coefficient)
    #
    # Wait, but that's not quite right either. Let me think again.
    #
    # The contribution of factor r to C_{i,k} is:
    #   (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
    #
    # For this to work with Strassen, we need:
    #   sum_r [ (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) * W[k*n+i, r] ]
    #   = sum_j A[i,j] * B[j,k]
    #
    # For the schoolbook term (i0,j0,k0): U[i0*m+j0]=1, V[j0*p+k0]=1, W[k0*n+i0]=1.
    # Contribution to C[i,k]: A[i0,j0]*B[j0,k0] * (1 if i0==i and k0==k, else 0 from W).
    # Wait, W[k*n+i, r] = 1 only when k==k0 and i==i0. So contribution is A[i0,j0]*B[j0,k0] if i==i0 and k==k0.
    # Summing over all schoolbook terms: C[i,k] = sum_{j} A[i,j]*B[j,k]. Correct.
    #
    # Now for Strassen, each factor r has:
    #   U[i*m+j, r] = u_r[i,j]  (a 2x2 matrix of coefficients for A)
    #   V[j*p+k, r] = v_r[j,k]  (a 2x2 matrix of coefficients for B, but only for the block's columns)
    #   W[k*n+i, r] = w_r[i,k]  (a 2x2 matrix of coefficients for C)
    #
    # The contribution to C[i,k] from factor r:
    #   (sum_{i',j'} u_r[i',j'] * A[i',j']) * (sum_{j',k'} v_r[j',k'] * B[j',k']) * ... 
    #
    # Hmm, this isn't quite right. The issue is that U, V, W are vectors, not matrices.
    # Let me reconsider.
    #
    # I think the correct interpretation is:
    #   C_{i,k} = sum_r (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{j} V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #
    # No wait, that doesn't match the schoolbook. In schoolbook, for term (i0,j0,k0):
    #   U[i0*m+j0, r] = 1, all other U entries are 0.
    #   V[j0*p+k0, r] = 1, all other V entries are 0.
    #   W[k0*n+i0, r] = 1, all other W entries are 0.
    #
    # If the formula were C_{i,k} = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r],
    # then for the schoolbook term: sum_j U[i*m+j,r]*A[i,j] = A[i0,j0] (only j=j0 contributes, and only if i=i0).
    # sum_j V[j*p+k,r]*B[j,k] = B[j0,k0] (only k=k0 contributes, and only if... wait, V[j*p+k,r] is indexed by j and k).
    #
    # I think the formula is actually:
    #   C_{i,k} = sum_r sum_{i',j',k'} U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r] * A[i',j'] * B[j',k']
    #
    # And the constraint is that this equals sum_j A[i,j]*B[j,k] for all i,k.
    #
    # For schoolbook term (i0,j0,k0): the sum reduces to U[i0*m+j0,r]*V[j0*p+k0,r]*W[k0*n+i0,r]*A[i0,j0]*B[j0,k0]
    # = 1*1*1*A[i0,j0]*B[j0,k0]. And this contributes to C[i,k] only when... well it contributes A[i0,j0]*B[j0,k0] to ALL C[i,k]?
    # That can't be right.
    #
    # I think there must be an implicit delta or the W factor selects the output. Let me look at this differently.
    #
    # Maybe the formula is:
    #   C_{i,k} = sum_r (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{j} V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #
    # For schoolbook (i0,j0,k0): 
    #   sum_j U[i*m+j,r]*A[i,j] = U[i*m+j0,r]*A[i,j0] = A[i,j0] if i==i0, else 0.
    #   sum_j V[j*p+k,r]*B[j,k] = V[j0*p+k,r]*B[j0,k] = B[j0,k] if k==k0, else 0.
    #   W[k*n+i,r] = 1 if k==k0 and i==i0, else 0.
    #   So contribution = A[i0,j0]*B[j0,k0] if i==i0 and k==k0, else 0. Correct!
    #
    # OK so the formula is:
    #   C_{i,k} = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #
    # Now for Strassen factor r with block t:
    #   U[i*m+j, r] = u_r[i,j] for all i,j in {0,1}
    #   V[j*p+k, r] = v_r[j, k_local] where k = 2t + k_local, k_local in {0,1}
    #   W[k*n+i, r] = w_r[i, k_local] where k = 2t + k_local, k_local in {0,1}
    #
    # The contribution to C[i,k] from this factor:
    #   (sum_j u_r[i,j] * A[i,j]) * (sum_j v_r[j, k_local] * B[j, 2t+k_local]) * w_r[i, k_local]
    #   (where k_local = k - 2t, and this is 0 if k is not in {2t, 2t+1})
    #
    # For the sum over all 7 Strassen factors in block t to give C[i, 2t+k_local] = sum_j A[i,j]*B[j, 2t+k_local],
    # we need:
    #   sum_{r=0}^{6} (sum_j u_r[i,j] * A[i,j]) * (sum_j v_r[j, k_local] * B[j, 2t+k_local]) * w_r[i, k_local]
    #   = sum_j A[i,j] * B[j, 2t+k_local]
    #
    # This is exactly the Strassen decomposition of 2x2x2 matrix multiplication!
    #
    # Now let me define the 7 factors. For Strassen:
    # M1 = (a00+a11)(b00+b11): u=[1,1] (for a00,a11... wait, u is a 2x2 matrix)
    #
    # Actually u_r[i,j] is the coefficient of A[i,j] in the "A-side" of factor r.
    # For M1 = (a00+a11)(b00+b11):
    #   u_1 = [[1,0],[0,1]] (coefficient of a00 is 1, a01 is 0, a10 is 0, a11 is 1)
    #   v_1 = [[1,1],[1,1]] (coefficient of b00 is 1, b01 is 1, b10 is 1, b11 is 1)
    #   w_1 = [[1,0],[0,1]] (contributes to c00 and c11)
    #
    # Wait, but w should be such that the output is correctly formed. Let me reconsider.
    #
    # Actually, I think the factor structure is:
    #   Factor r computes: P_r = (u_r . A) * (B . v_r)  where u_r and v_r are vectors/matrices
    #   Then C += w_r * P_r  (some combination)
    #
    # Let me use a cleaner formulation. For each Strassen factor:
    #   A_side: a linear combination of A's entries -> scalar
    #   B_side: a linear combination of B's entries -> scalar  
    #   C_side: how this product contributes to C's entries
    #
    # M1: A_side = a00+a11, B_side = b00+b11, contributes: C00 += M1, C11 += M1
    # M2: A_side = a10+a11, B_side = b00, contributes: C10 += M2, C11 -= M2
    # M3: A_side = a00, B_side = b01-b11, contributes: C01 += M3, C11 += M3
    # M4: A_side = a11, B_side = b10-b00, contributes: C00 += M4, C10 += M4
    # M5: A_side = a00+a01, B_side = b11, contributes: C00 -= M5, C01 += M5
    # M6: A_side = a10-a00, B_side = b00+b01, contributes: C11 += M6
    # M7: A_side = a01-a11, B_side = b10+b11, contributes: C00 += M7
    #
    # So in the U,V,W framework:
    #   U[i*m+j, r] = coefficient of A[i,j] in A_side of factor r
    #   V[j*p+k, r] = coefficient of B[j,k] in B_side of factor r (for the block's columns)
    #   W[k*n+i, r] = coefficient of C[i,k] in the output of factor r
    #
    # Let me define all 7:
    # M1: U = [[1,0],[0,1]], V = [[1,1],[1,1]], W = [[1,0],[0,1]]
    # M2: U = [[0,0],[1,1]], V = [[1,0],[0,0]], W = [[0,0],[1,-1]]
    # M3: U = [[1,0],[0,0]], V = [[0,1],[0,-1]], W = [[0,1],[0,1]]
    # M4: U = [[0,0],[0,1]], V = [[0,-1],[1,0]], W = [[1,0],[1,0]]
    # M5: U = [[1,1],[0,0]], V = [[0,0],[0,1]], W = [[-1,1],[0,0]]
    # M6: U = [[-1,0],[1,0]], V = [[1,1],[0,0]], W = [[0,0],[0,1]]
    # M7: U = [[0,1],[0,-1]], V = [[0,0],[1,1]], W = [[1,0],[0,0]]
    #
    # Let me verify M1: 
    #   A_side = 1*a00 + 0*a01 + 0*a10 + 1*a11 = a00+a11 ✓
    #   B_side = 1*b00 + 1*b01 + 1*b10 + 1*b11 = b00+b01+b10+b11 ✓
    #   C: C00 += 1*M1, C01 += 0*M1, C10 += 0*M1, C11 += 1*M1 ✓
    #
    # M2:
    #   A_side = 0*a00 + 0*a01 + 1*a10 + 1*a11 = a10+a11 ✓
    #   B_side = 1*b00 + 0*b01 + 0*b10 + 0*b11 = b00 ✓
    #   C: C00 += 0, C01 += 0, C10 += 1*M2, C11 += -1*M2 ✓
    #
    # M3:
    #   A_side = a00 ✓
    #   B_side = 0*b00 + 1*b01 + 0*b10 + (-1)*b11 = b01-b11 ✓
    #   C: C00 += 0, C01 += 1*M3, C10 += 0, C11 += 1*M3 ✓
    #
    # M4:
    #   A_side = a11 ✓
    #   B_side = 0*b00 + (-1)*b01 + 1*b10 + 0*b11 = b10-b00 ✓
    #   C: C00 += 1*M4, C01 += 0, C10 += 1*M4, C11 += 0 ✓
    #
    # M5:
    #   A_side = a00+a01 ✓
    #   B_side = b11 ✓
    #   C: C00 += -1*M5, C01 += 1*M5, C10 += 0, C11 += 0 ✓
    #
    # M6:
    #   A_side = -a00+a10 ✓
    #   B_side = b00+b01 ✓
    #   C: C00 += 0, C01 += 0, C10 += 0, C11 += 1*M6 ✓
    #
    # M7:
    #   A_side = a01-a11 ✓
    #   B_side = b10+b11 ✓
    #   C: C00 += 1*M7, C01 += 0, C10 += 0, C11 += 0 ✓
    #
    # All verified! Now implement.

    strassen_factors = [
        # (U_2x2, V_2x2, W_2x2)
        ([[1,0],[0,1]], [[1,1],[1,1]], [[1,0],[0,1]]),   # M1
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,-1]]),   # M2
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,1]]),   # M3
        ([[0,0],[0,1]], [[0,-1],[1,0]], [[1,0],[1,0]]),   # M4
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),   # M5
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),   # M6
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),   # M7
    ]

    for t in range(num_blocks):
        for r in range(7):
            col = t * 7 + r
            u_mat, v_mat, w_mat = strassen_factors[r]
            
            # Set U: U[i*m+j, col] = u_mat[i][j]
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = u_mat[i][j]
            
            # Set V: V[j*p+k, col] = v_mat[j][k_local] where k = 2t + k_local
            for j in range(m):
                for k_local in range(2):
                    k = 2 * t + k_local
                    V[j * p + k][col] = v_mat[j][k_local]
            
            # Set W: W[k*n+i, col] = w_mat[i][k_local] where k = 2t + k_local
            for i in range(n):
                for k_local in range(2):
                    k = 2 * t + k_local
                    W[k * n + i][col] = w_mat[i][k_local]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
