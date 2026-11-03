def construct():
    n, m, p = 3, 5, 6
    # Use decomposition: for each row i of A, compute A_i @ B
    # Each such subproblem has rank m = 5
    # Total rank = n * m = 15
    rank = n * m  # 15
    
    U = [[0] * rank for _ in range(n * m)]  # shape (3*5, 15) = (15, 15)
    V = [[0] * rank for _ in range(m * p)]  # shape (5*6, 15) = (30, 15)
    W = [[0] * rank for _ in range(p * n)]  # shape (6*3, 15) = (18, 15)
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            # U[i*m + j, c] = 1 (A[i,j] coefficient)
            U[i * m + j][c] = 1
            # V[j*p + k, c] = 1 for all k? No, that's not right.
            
    # Actually, let me think about this more carefully.
    # The bilinear form is sum_j A[i,j] * B[j,k] for each (i,k).
    # For a rank-1 term with index c:
    #   U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j,k should give the identity.
    # 
    # For the "row-wise" decomposition:
    # For each i, j: one term where
    #   U[i*m+j, c] = 1
    #   V[j*p+k, c] = 1 for the specific k? No, that doesn't work.
    #
    # Let me use the standard bilinear decomposition.
    # Actually, the simplest is: for each (i, j, k) triple, one term. That's rank 90.
    # 
    # To reduce: for fixed i, the subproblem is M_{1,m,p} with rank m.
    # The decomposition: for j in 0..m-1, 
    #   U[i*m+j, c_ij] = 1
    #   V[j*p+k, c_ij] = 1 for all k? No.
    #
    # I think I need to be more careful. Let me just use rank 15 with the following:
    # For each i (0..2) and j (0..4), create one term c = i*5 + j
    # U[i*5+j, c] = 1
    # V[j*6+k, c] = delta_{k, ?} -- this doesn't work because V needs to depend on k.
    #
    # The issue: each rank-1 term contributes U_val * V_val * W_val to T[i,j,k].
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k (since T[i,j,k]=1 as bilinear coefficient, but actually the tensor is T[i,k,j] = A[i,j]B[j,k], so the structural tensor has S[i,j,k] = 1).
    #
    # So we need: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    #
    # If we set for each (i,j): c = i*m+j
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1 for all k -- but then V has 1s in row j*p+k for all k, same c.
    # W[k*n+i, c] = 1 for all k
    # Then sum_c U[i*m+j,c]V[j*p+k,c]W[k*n+i,c] = sum over c' where U[i*m+j,c']=1... only c=i*m+j has U=1.
    # Then product = 1 * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    # If we set V[j*p+k, i*m+j] = 1 and W[k*n+i, i*m+j] = 1, then product = 1.
    # But V[j*p+k, i*m+j] is the same for all i! So V[j*p+k, c] must be 1 for all c corresponding to any i.
    # Similarly W[k*n+i, c] must be 1 for all c corresponding to any i.
    #
    # So set: for all j, k, and all i: V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1.
    # But V has only m*p = 30 rows and rank=15 columns. Column c = i*m+j.
    # For fixed j, k: V[j*p+k, c] = 1 if c = i*m+j for some i. That is 3 values of i.
    # So V[j*p+k, c] = 1 for c in {0*m+j, 1*m+j, 2*m+j} = {j, 5+j, 10+j}.
    # Similarly W[k*n+i, c] = 1 for c in {0*m+i, 1*m+i, 2*m+i} = {i, 5+i, 10+i}.
    #
    # Then for given (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # U[i*m+j,c] = 1 only when c = i*m+j.
    # So only c = i*m+j contributes.
    # V[j*p+k, i*m+j] = 1 (by construction, since c=i*m+j is in {j, 5+j, 10+j}).
    # W[k*n+i, i*m+j] = 1 (by construction, since c=i*m+j is in {i, 5+i, 10+i}).
    # So product = 1*1*1 = 1.
    #
    # But wait, is this correct? Let's verify the tensor entry.
    # The bilinear form is C[i,k] = sum_j A[i,j] B[j,k].
    # In the decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_k' V[j*p+k',c] B[j,k']) * W[k*n+i,c]
    # Actually, the convention is: C[i,k] = sum_c u_c(A) * v_c(B) * W[k*n+i,c]
    # where u_c(A) = sum_{i,j} U[i*m+j,c] A[i,j] and v_c(B) = sum_{j,k} V[j*p+k,c] B[j,k].
    #
    # With our assignment:
    # u_c(A) = sum_{i',j'} U[i'*m+j',c] A[i',j']
    # For c = i0*m+j0, U[i0*m+j0, c] = 1, all others 0. So u_c(A) = A[i0,j0].
    # v_c(B) = sum_{j',k'} V[j'*p+k',c] B[j',k']
    # For c = i0*m+j0, V[j'*p+k',c] = 1 when j' = j0 (and any k'), because c=i0*m+j0 implies the j-component is j0.
    # So v_c(B) = sum_{k'} B[j0,k'] = sum of row j0 of B.
    # W[k*n+i,c] = 1 when the i-component of c is i, i.e., when c = i*m+j0 for any j. Since c = i0*m+j0, this is 1 when i = i0.
    # So W[k*n+i,c] = 1 if i=i0, else 0.
    #
    # Then C[i,k] = sum_c u_c v_c W[k*n+i,c] = sum_{i0,j0} A[i0,j0] * (sum_{k'} B[j0,k']) * delta_{i,i0}
    # = A[i,j0] * (sum_{k'} B[j0,k']) summed over j0? No, c = i0*m+j0, so:
    # = sum_{j0} A[i,j0] * (sum_{k'} B[j0,k']) 
    # This is NOT equal to sum_{j0} A[i,j0] B[j0,k]! It sums over k' instead of fixing k.
    #
    # So this decomposition is WRONG. I made an error.

    # The correct approach: We need V[j*p+k, c] to be 1 only for specific k.
    # But c is indexed by (i,j) in my attempt. To fix k, I need c to depend on k too.
    # So the naive rank is n*m*p = 90.
    #
    # To reduce rank, we need to exploit bilinear structure.
    # For M_{1,5,6}: rank is 5. The decomposition:
    # For j=0..4, c=j:
    #   U[0*5+j, c] = 1  (only row 0 of A)
    #   V[j*6+k, c] = 1 for all k  -> same problem!
    #
    # I think the rank of M_{1,m,n} is actually m, but the decomposition requires:
    # u_j(A) = A[j], v_j(B) = B[j, :] (row j), and then C[k] = sum_j A[j] B[j,k].
    # In the tensor form: T[1,1,m] has T[1,k,j] = 1 for all k,j? No.
    # The structural tensor for 1×m×n: S[i,k,j] with i in 1, k in n, j in m. S[1,k,j] = 1.
    # Rank-1 decomposition: sum_c U[1*m+j? No, U has dimension (1*m) = m.
    # U[j, c] * V[j*p+k, c] * W[k*1+1, c]
    # For c = j': U[j, c] = delta_{j,j'}, V[j'*p+k, c] = ?, W[k*1+1, c] = ?
    # We need sum_c U[j,c] V[j'*p+k,c] W[k,c] = 1 for all j,k (S[1,k,j]=1).
    # Wait, indices: S[i,k,j] = 1. Decomposition: sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = S[i,k,j] = 1.
    # But the summation is over c, and the indices j, k, i are fixed in S[i,k,j].
    # So: for all i, k, j: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    #
    # For M_{1,5,6}: i fixed to 0. S[0,k,j] = 1 for all k,j.
    # Need: for all k, j: sum_c U[0*5+j, c] V[j*6+k, c] W[k*1+0, c] = 1.
    #
    # If we use c = j (0..4):
    # U[0*5+j, c'] = delta_{j,c'}
    # Then for fixed j,k: sum_c delta_{j,c} V[j*6+k, c] W[k, c] = V[j*6+k, j] W[k, j].
    # We need this = 1 for all k, j.
    # So set V[j*6+k, j] = 1 and W[k, j] = 1 for all k, j.
    # But V[j*6+k, j] is the same for all k. So V has 1s in column j at rows j*6+0, j*6+1, ..., j*6+5.
    # W[k, j] = 1 for all k, j. W has 6 rows, 5 columns.
    #
    # Check: for M_{1,5,6}, with this:
    # sum_c U[5+j, c] V[j*6+k, c] W[k+0, c] 
    # = sum_c delta_{j,c} V[j*6+k, c] W[k, c]
    # = V[j*6+k, j] W[k, j] = 1 * 1 = 1. Correct!
    #
    # Now for full 3×5×6:
    # Can we share the V and W across i?
    # For i=0,1,2, we need separate U indices.
    # Let c = i*5 + j for i in 0,1,2, j in 0,4.
    # U[i*5+j, c] = 1 only when c = i*5+j.
    # V[j*6+k, c] = 1 when the j-component of c is j, i.e., when c mod 5 = j. So V[j*6+k, c] = 1 if c mod 5 = j.
    # W[k*3+i, c] = 1 when the i-component of c is i, i.e., when c div 5 = i. So W[k*3+i, c] = 1 if c // 5 = i.
    #
    # Check: for all i, j, k:
    # sum_c U[i*5+j, c] V[j*6+k, c] W[k*3+i, c]
    # U[i*5+j, c] = 1 only if c = i*5+j.
    # So only c = i*5+j contributes.
    # V[j*6+k, i*5+j] = 1 iff (i*5+j) mod 5 = j iff i*5 mod 5 = 0 = 0. Yes, always 1.
    # W[k*3+i, i*5+j] = 1 iff (i*5+j) // 5 = i iff i = i. Yes, always 1.
    # So product = 1. Correct!
    #
    # Rank = 15. Let me implement this.

    n, m, p = 3, 5, 6
    rank = n * m  # 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            # V[j*p+k, c] = 1 for all k if c mod m == j
            for k in range(p):
                if c % m == j:
                    V[j * p + k][c] = 1
            # W[k*n+i, c] = 1 for all k if c // m == i
            for k in range(p):
                if c // m == i:
                    W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
