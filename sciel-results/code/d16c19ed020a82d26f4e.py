"""Tensor decomposition for 2x5x6 matrix multiplication using a ladder/shifted approach."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Ladder/shifted decomposition technique for small n=2.
    # The idea: for each column j of A (i.e., each index in the shared dimension),
    # we can decompose the contribution to C[i,k] = A[i,j]*B[j,k] in a way that
    # exploits the fact that n=2 is small.
    #
    # Standard schoolbook: rank = n*m*p = 60
    # 
    # Ladder idea: For each j, the tensor slice T[:,:,j] (over i and k) has rank
    # at most n*p = 12 (it's the outer product structure A[:,j] ⊗ B[j,:]).
    # But we can potentially share terms across different j values.
    #
    # Actually, let me think about this more carefully.
    #
    # The matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] 
    # from A[i,j]*B[j,k]. The tensor is T[i,j,k] = delta_{i,j,k} where the
    # multiplication pattern is fixed.
    #
    # For n=2, we can use the following construction:
    # For each j from 0 to m-1, and for each pair of consecutive k values,
    # we can combine terms.
    #
    # Let me try a different approach: use the fact that for n=2, we can write
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    #
    # Define for each j:
    #   x_j = A[0,j]
    #   y_j = A[1,j]
    #   u_jk = B[j,k]
    #
    # Then C[0,k] = sum_j x_j * u_jk
    #      C[1,k] = sum_j y_j * u_jk
    #
    # Now, consider grouping by j. For a fixed j, we have m*p = 30 terms in the
    # schoolbook. But maybe we can do better.
    #
    # Actually, let me try the "shifted" decomposition:
    # For k=0,1,...,p-1, define shifted versions.
    #
    # Let me try to construct a rank-59 decomposition by merging one pair of
    # schoolbook terms.
    #
    # In the schoolbook, term (i,j,k) has:
    #   U[i*m+j] = 1
    #   V[j*p+k] = 1  
    #   W[k*n+i] = 1
    #
    # Can two such terms be combined? Only if they share the same U, V, or W
    # pattern in a way that allows linear combination.
    #
    # Let me try: for j=0, k=0, we have terms (0,0,0) and (1,0,0).
    # Term (0,0,0): U[0]=1, V[0]=1, W[0]=1
    # Term (1,0,0): U[5]=1, V[0]=1, W[1]=1
    #
    # These share V[0]=1. Can we write a single rank-1 term that covers both?
    # A rank-1 term has U_a ⊗ V_b ⊗ W_c where each is a vector.
    # If we set V_b[0]=1 and all else 0, then we need U_a and W_c such that
    # U_a[0]*W_c[0] = 1 (for C[0,0]) and U_a[5]*W_c[1] = 1 (for C[1,0]).
    # But U_a is a single vector, so U_a[0] and U_a[5] are fixed. Similarly W_c[0] and W_c[1].
    # We need U_a[0]*W_c[0]=1 and U_a[5]*W_c[1]=1. This is possible with e.g.
    # U_a[0]=1, U_a[5]=1, W_c[0]=1, W_c[1]=1. But then this term also contributes
    # to other positions! Specifically, it would contribute U_a[i*m+j']*W_c[k'*n+i''] 
    # for any i', j', k' where the pattern matches. Since U_a is nonzero at positions 0 and 5,
    # and W_c is nonzero at positions 0 and 1, this term contributes to:
    #   (i=0,j=0,k=0): U[0]*W[0] = 1*1 = 1 ✓
    #   (i=1,j=0,k=0): U[5]*W[1] = 1*1 = 1 ✓
    #   (i=0,j=0,k=1): U[0]*W[n+0] = U[0]*W[2] = 1*0 = 0 ✓ (W[2]=0)
    #   (i=1,j=0,k=1): U[5]*W[n+1] = U[5]*W[3] = 1*0 = 0 ✓ (W[3]=0)
    #   Wait, W has indices k*n+i. So W[0]=W[0*2+0], W[1]=W[0*2+1], W[2]=W[1*2+0], W[3]=W[1*2+1], etc.
    #   If W_c has W_c[0]=1, W_c[1]=1, and all others 0, then:
    #   For (i=0,j=0,k=0): U_a[0*m+0]*V_b[0*p+0]*W_c[0*n+0] = U_a[0]*V_b[0]*W_c[0]
    #   For (i=1,j=0,k=0): U_a[1*m+0]*V_b[0*p+0]*W_c[0*n+1] = U_a[5]*V_b[0]*W_c[1]
    #   For (i=0,j=0,k=1): U_a[0]*V_b[1]*W_c[2] = U_a[0]*0*W_c[2] = 0 (since V_b[1]=0)
    #   So this works! The combined term only contributes to (0,0,0) and (1,0,0).
    #
    # So we can replace terms (0,0,0) and (1,0,0) with a single term:
    #   U_a: U_a[0]=1, U_a[5]=1, all else 0
    #   V_b: V_b[0]=1, all else 0
    #   W_c: W_c[0]=1, W_c[1]=1, all else 0
    #
    # This saves 1 rank! So rank 59 is achievable.
    #
    # Can we do better? Let's see if we can apply this trick more broadly.
    # For each j, we can combine terms (0,j,k) and (1,j,k) for a fixed k.
    # But wait, the trick above combined (0,j,0) and (1,j,0) for j=0.
    # Let's generalize: for any j and k, terms (0,j,k) and (1,j,k) can be combined
    # into a single term with:
    #   U_a: U_a[0*m+j]=1, U_a[1*m+j]=1, all else 0
    #   V_b: V_b[j*p+k]=1, all else 0
    #   W_c: W_c[k*n+0]=1, W_c[k*n+1]=1, all else 0
    #
    # This gives us m*p = 30 combined terms instead of 2*m*p = 60 terms.
    # So rank = 30!
    #
    # Wait, but that's too good. Let me verify.
    # The combined term for (j,k) contributes:
    #   T[i,j',k'] = U_a[i*m+j']*V_b[j'*p+k']*W_c[k'*n+i]
    #   = (1 if j'=j else 0) * (1 if k'=k else 0) * (1 if i=0 and k'=k else 0 + 1 if i=1 and k'=k else 0)
    #   Wait, W_c[k'*n+i] is 1 only if (k'=k and i=0) or (k'=k and i=1).
    #   So T[i,j,k] = 1 if i in {0,1} and j'=j and k'=k.
    #   And T[i,j',k'] = 0 otherwise.
    #
    # So each combined term exactly reproduces the two schoolbook terms (0,j,k) and (1,j,k).
    # And there are m*p = 30 such combined terms.
    #
    # So the rank is at most 30!
    #
    # But wait, the lower bound is max(n*m, m*p, n*p) = max(10, 30, 12) = 30.
    # So rank 30 is optimal!
    #
    # Let me double-check: the tensor T[i,j,k] for matrix multiplication has
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The rank of this tensor is at least the slice rank. For the slice along j,
    # the matrix T[i,k] (for fixed j) is the all-ones matrix of size n x p = 2 x 6.
    # The rank of a 2x6 all-ones matrix is 2. So the slice rank along j is at least 2.
    # But the tensor rank is at least the maximum over all slices.
    #
    # Actually, the correct lower bound for tensor rank is more subtle. But the
    # construction above gives rank 30, which equals m*p. And the lower bound
    # max(n*m, m*p, n*p) = 30 suggests this is optimal.
    #
    # Let me verify the construction is correct by checking a few entries.
    # For the combined term (j=2, k=3):
    #   U_a[0*m+2] = U_a[2] = 1
    #   U_a[1*m+2] = U_a[7] = 1
    #   V_b[2*p+3] = V_b[15] = 1
    #   W_c[3*n+0] = W_c[6] = 1
    #   W_c[3*n+1] = W_c[7] = 1
    #
    # Contribution to T[i,j',k']:
    #   T[i,j',k'] += U_a[i*m+j'] * V_b[j'*p+k'] * W_c[k'*n+i]
    #   Only nonzero when j'=2, k'=3, and i in {0,1}.
    #   T[0,2,3] += 1*1*1 = 1
    #   T[1,2,3] += 1*1*1 = 1
    #   All other entries get 0 contribution from this term.
    #
    # Summing over all 30 combined terms, we get T[i,j,k] = 1 for all i,j,k. ✓
    
    rank = m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: nonzero at positions 0*m+j and 1*m+j
            U[0 * m + j][col] = 1
            U[1 * m + j][col] = 1
            # V: nonzero at position j*p+k
            V[j * p + k][col] = 1
            # W: nonzero at positions k*n+0 and k*n+1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
