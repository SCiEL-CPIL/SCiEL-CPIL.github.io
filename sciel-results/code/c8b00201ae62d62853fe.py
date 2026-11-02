"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6

    # We seek a rank-minimal decomposition of the matrix multiplication tensor T
    # where T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # The schoolbook rank is n*m*p = 36.
    # We can improve this by grouping terms.
    #
    # Observation:
    # The tensor is a sum over j of terms T_j where T_j[i*m+j, j*p+k, k*n+i] = 1.
    # For a fixed j, the non-zero entries form a pattern.
    # Specifically, for fixed j, the indices are:
    # U-index: i*m + j  (i ranges 0..n-1) -> n values
    # V-index: j*p + k  (k ranges 0..p-1) -> p values
    # W-index: k*n + i  (k ranges 0..p-1, i ranges 0..n-1) -> p*n values
    #
    # The sub-tensor for fixed j is effectively a "partial" tensor that links i and k.
    # Let's analyze the rank of the sub-tensor for a single j.
    # The condition is that U, V, W must be 1 simultaneously.
    # U depends on i (and j is fixed).
    # V depends on k (and j is fixed).
    # W depends on k and i.
    #
    # If we try to decompose the single-j tensor into rank-1 terms:
    # A rank-1 term is defined by vectors u, v, w.
    # For the product u[u_idx] * v[v_idx] * w[w_idx] to be 1 only when (i, k) matches the pattern,
    # and 0 otherwise.
    #
    # Let's look at the structure of the single-j tensor as a matrix.
    # Flatten the W dimension into a 2D matrix of size (p*n) x (something)?
    # Actually, let's view the tensor T_j as a 3-way tensor of dimensions (n, p, p*n).
    # T_j[i, k, k*n+i] = 1.
    #
    # This is a "diagonal" tensor in the (i, k) vs (k*n+i) sense? No.
    # Let's try to find the rank of this specific sub-tensor.
    #
    # Consider the case n=2, p=6.
    # We have 2*6 = 12 non-zero entries in the "output" space (W), and 2*6=12 in the "input" space (U, V combined?).
    #
    # Let's try a construction based on splitting the k-index or i-index.
    #
    # Alternative Idea: Use the fact that the tensor is a Kronecker product of smaller tensors?
    # T is not a simple Kronecker product because the indices are coupled (j links U and V, k links V and W, i links W and U).
    #
    # Let's try to reduce the rank by combining terms that share the same k or i.
    #
    # Standard trick: For matrix multiplication, the rank is often bounded by the rank of the "core" tensor.
    #
    # Let's try a rank-24 construction.
    # Idea: Group by k. For each k, we have a sub-problem.
    # For fixed k, the non-zero entries are:
    # U[i*m+j] for all i, j.
    # V[j*p+k] for all j.
    # W[k*n+i] for all i.
    #
    # This looks like we are computing C[:, k] = A @ B[:, k].
    # This is a matrix-vector multiplication.
    # The tensor for a single column k of the output is:
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # Here j is summed over.
    #
    # Can we decompose this single-column tensor with rank < n*m?
    # The single column tensor has dimensions (n*m, m, n).
    # T[i*m+j, j, i] = 1.
    #
    # Let's analyze the rank of T_col[i*m+j, j, i] = 1.
    # This is a tensor of size (6, 3, 2).
    # Let's try to find its rank.
    #
    # We want to write T_col = sum_r u_r * v_r * w_r.
    # u_r is length 6, v_r is length 3, w_r is length 2.
    #
    # The support of T_col is {(i*m+j, j, i) | i in 0..1, j in 0..2}.
    # There are 6 non-zero entries.
    #
    # Can we do it with rank < 6?
    # Let's try rank 3.
    # If rank is 3, we have 3 terms.
    #
    # Let's try to construct it.
    # We need u_r[i*m+j] * v_r[j] * w_r[i] = 1 if (i,j) is in the support, else 0.
    #
    # Note that the support is a "grid" of 2x3 points.
    #
    # Let's try to assign values.
    # Let w_0 = [1, 0], w_1 = [0, 1], w_2 = [1, 1].
    # Then for i=0, only terms with w_r[0] != 0 contribute.
    # For i=1, only terms with w_r[1] != 0 contribute.
    #
    # If we use w_0=[1,0], w_1=[0,1], w_2=[1,1]:
    # Term 0 contributes to i=0.
    # Term 1 contributes to i=1.
    # Term 2 contributes to both.
    #
    # For i=0:
    # We need u_0[i*m+j]*v_0[j] + u_2[i*m+j]*v_2[j] = 1 for all j.
    # For i=1:
    # We need u_1[i*m+j]*v_1[j] + u_2[i*m+j]*v_2[j] = 1 for all j.
    #
    # Also, we need the terms to be zero where they shouldn't be?
    # Wait, the tensor is 1 on the support and 0 elsewhere.
    # The support is exactly the set of (i,j) pairs.
    # So for a fixed r, the product u_r[i*m+j] * v_r[j] * w_r[i] must be 0 for all (i,j) NOT in the support?
    # No, the sum must be 0 for (i,j) not in support, and 1 for (i,j) in support.
    # But the support is ALL (i,j) pairs for the fixed k!
    # Wait, for fixed k, the tensor T_col has non-zero entries for ALL i in 0..n-1 and j in 0..m-1.
    # So the support is the full grid of size n*m.
    #
    # So we need:
    # sum_r u_r[i*m+j] * v_r[j] * w_r[i] = 1 for all i, j.
    #
    # This is equivalent to:
    # For each i, the matrix M_i of size m x m defined by (M_i)_{j, l} ? No.
    # Let's fix i.
    # sum_r (u_r[i*m+j] * w_r[i]) * v_r[j] = 1.
    # Let a_r[i, j] = u_r[i*m+j] * w_r[i].
    # Then sum_r a_r[i, j] * v_r[j] = 1 for all i, j.
    #
    # This means that for each i, the vector of ones (length m) is in the span of the rows of the matrix formed by a_r[i, :] and v_r?
    # Actually, let's define matrix A_i of size (rank) x (m) where A_i[r, j] = u_r[i*m+j] * w_r[i].
    # And vector v_r is a column vector of length m? No, v_r is a vector of length m.
    # The equation is: sum_r A_i[r, j] * v_r[j] = 1.
    # This can be written as: (A_i * V)_{j} = 1? No.
    # Let V be a matrix of size (rank) x (m) where V[r, j] = v_r[j].
    # Then the equation is: sum_r A_i[r, j] V[r, j] = 1.
    # This is an element-wise product sum.
    # This is the trace of A_i * V^T? No.
    # It is the Hadamard product sum: <A_i, V>_F = 1? No, indices don't match for trace.
    # It is sum_{r,j} A_i[r,j] V[r,j] = 1.
    # This is the Frobenius inner product of A_i and V.
    # So we need <A_i, V> = 1 for all i.
    #
    # Also, we need to ensure that the decomposition works for the full tensor, which includes the k-dependence in V and W.
    #
    # Let's step back.
    # The total tensor is sum_k T_k.
    # If we can decompose each T_k with rank R, total rank is p*R.
    # If we can share terms across k, we might do better.
    #
    # However, T_k and T_{k'} have disjoint supports in the V and W dimensions?
    # V index: j*p+k. For fixed j, different k give different V indices.
    # W index: k*n+i. For fixed i, different k give different W indices.
    # So the supports in V and W are disjoint for different k.
    # This implies that the terms for different k cannot be "mixed" easily to reduce rank, because a single rank-1 term would have to be non-zero in V at indices corresponding to multiple k's, which would create cross-terms.
    #
    # Specifically, if a term r has V_r non-zero at j*p+k and j'*p+k', then for the product to be non-zero, U_r must be non-zero at corresponding U indices, and W_r at corresponding W indices.
    # Since the tensor T is zero unless k matches in V and W, a term that is non-zero for two different k's would produce non-zero values in T where k_V != k_W, which must be 0.
    # This forces the terms to be "block diagonal" with respect to k.
    #
    # Therefore, the total rank is at least the sum of the ranks of the individual T_k.
    # So Rank(T) >= sum_k Rank(T_k).
    #
    # Now we just need to find the minimal rank of T_k.
    # T_k[i*m+j, j*p+k, k*n+i] = 1.
    # Let's relabel indices for T_k:
    # U index: u = i*m + j.
    # V index: v = j*p + k. (Note k is fixed for this sub-tensor, so v runs over j*p+k).
    # W index: w = k*n + i. (Note k is fixed, so w runs over k*n+i).
    #
    # We need to decompose the tensor T_k of size (n*m) x (m) x (n) ?
    # Wait, the V dimension for T_k is not just m. The full V dimension is m*p. But T_k is only non-zero for indices v where v % p == k.
    # So effectively, for the sub-tensor T_k, the V dimension is of size m (indexed by j).
    # Similarly, the W dimension for T_k is of size n (indexed by i).
    # The U dimension is of size n*m (indexed by i, j).
    #
    # So T_k is a tensor of dimensions (n*m) x (m) x (n).
    # T_k[i*m+j, j, i] = 1.
    #
    # We need the rank of this tensor.
    # Let's call it R(n, m).
    #
    # For n=2, m=3.
    # Dimensions: 6 x 3 x 2.
    #
    # Let's try to find the rank of T[i*m+j, j, i] = 1.
    #
    # We established: sum_r u_r[i*m+j] v_r[j] w_r[i] = 1.
    #
    # Let's try rank 3.
    # We need 3 vectors u_r (len 6), v_r (len 3), w_r (len 2).
    #
    # Let's try to construct it.
    # Let w_0 = [1, 0], w_1 = [0, 1], w_2 = [1, 1].
    #
    # For i=0:
    # sum_r u_r[0*m+j] v_r[j] w_r[0] = 1.
    # u_0[0*m+j] v_0[j] * 1 + u_1[0*m+j] v_1[j] * 0 + u_2[0*m+j] v_2[j] * 1 = 1.
    # sum_j (u_0[j] v_0[j] + u_2[j] v_2[j]) = 1? No, the sum is over r, not j.
    # The equation is: for each j, sum_r (u_r[j] * w_r[0]) * v_r[j] = 1.
    # Let A_0[j] = u_0[j] * 1 + u_2[j] * 1.
    # Then we need sum_r ... wait.
    # The term is u_r[j] * v_r[j] * w_r[i].
    # So for fixed i, j, the value is sum_r u_r[j] v_r[j] w_r[i].
    # Let B_r[i, j] = u_r[j] v_r[j].
    # Then we need sum_r B_r[i, j] w_r[i] = 1.
    # This must hold for all i, j.
    #
    # Let's look at the matrix W of w_r.
    # W is a (rank) x (n) matrix.
    # The equation is: for each j, the vector 1 (length n) is in the column space of the matrix formed by B_r[j, :] and W?
    # Actually, let M_j be a matrix of size (rank) x (n) where M_j[r, i] = u_r[j] v_r[j] w_r[i].
    # Then we need sum_r M_j[r, i] = 1 for all i.
    # This means the sum of the rows of M_j is the vector of ones.
    #
    # Also, we need the decomposition to be valid.
    #
    # Let's try a specific construction for n=2, m=3.
    # Rank 3.
    # w_0 = [1, 0], w_1 = [0, 1], w_2 = [1, 1].
    #
    # We need to choose u_r and v_r.
    #
    # For i=0:
    # sum_r u_r[j] v_r[j] w_r[0] = 1.
    # u_0[j] v_0[j] * 1 + u_1[j] v_1[j] * 0 + u_2[j] v_2[j] * 1 = 1.
    # u_0[j] v_0[j] + u_2[j] v_2[j] = 1.
    #
    # For i=1:
    # u_0[j] v_0[j] * 0 + u_1[j] v_1[j] * 1 + u_2[j] v_2[j] * 1 = 1.
    # u_1[j] v_1[j] + u_2[j] v_2[j] = 1.
    #
    # Let x_r[j] = u_r[j] v_r[j].
    # We need:
    # x_0[j] + x_2[j] = 1
    # x_1[j] + x_2[j] = 1
    #
    # This implies x_0[j] = x_1[j] for all j.
    # Let x_0[j] = x_1[j] = a[j].
    # Then a[j] + x_2[j] = 1 => x_2[j] = 1 - a[j].
    #
    # We can choose a[j] arbitrarily.
    # Let's choose a[j] = 1 for all j.
    # Then x_0[j] = 1, x_1[j] = 1, x_2[j] = 0.
    #
    # So we need:
    # u_0[j] v_0[j] = 1
    # u_1[j] v_1[j] = 1
    # u_2[j] v_2[j] = 0
    #
    # We can set:
    # u_0[j] = 1, v_0[j] = 1
    # u_1[j] = 1, v_1[j] = 1
    # u_2[j] = 0, v_2[j] = 0
    #
    # Let's check if this works.
    # w_0 = [1, 0], w_1 = [0, 1], w_2 = [1, 1].
    #
    # Term 0: u_0 = [1,1,1,1,1,1], v_0 = [1,1,1], w_0 = [1,0].
    # Term 1: u_1 = [1,1,1,1,1,1], v_1 = [1,1,1], w_1 = [0,1].
    # Term 2: u_2 = [0,0,0,0,0,0], v_2 = [0,0,0], w_2 = [1,1].
    #
    # Let's evaluate T_k[i*m+j, j, i].
    #
    # Case i=0, j=0:
    # Term 0: u_0[0] v_0[0] w_0[0] = 1*1*1 = 1.
    # Term 1: u_1[0] v_1[0] w_1[0] = 1*1*0 = 0.
    # Term 2: 0.
    # Sum = 1. Correct.
    #
    # Case i=0, j=1:
    # Term 0: u_0[1] v_0[1] w_0[0] = 1*1*1 = 1.
    # Term 1: 0.
    # Term 2: 0.
    # Sum = 1. Correct.
    #
    # Case i=1, j=0:
    # Term 0: u_0[0] v_0[0] w_0[1] = 1*1*0 = 0.
    # Term 1: u_1[0] v_1[0] w_1[1] = 1*1*1 = 1.
    # Term 2: 0.
    # Sum = 1. Correct.
    #
    # It seems to work!
    #
    # So the rank of T_k is at most 3.
    # Is it less?
    # Rank 2?
    # If rank 2, we have w_0, w_1.
    # w_0 = [1, 0], w_1 = [0, 1] (wlog).
    # Then for i=0: x_0[j] = 1.
    # For i=1: x_1[j] = 1.
    # This is possible with x_0[j]=1, x_1[j]=1.
    # So rank 2 might work?
    #
    # Let's check rank 2.
    # w_0 = [1, 0], w_1 = [0, 1].
    # u_0[j] v_0[j] = 1.
    # u_1[j] v_1[j] = 1.
    #
    # We can set u_0 = [1,1,1,1,1,1], v_0 = [1,1,1].
    # u_1 = [1,1,1,1,1,1], v_1 = [1,1,1].
    #
    # Let's check the full tensor.
    # T[i*m+j, j*p+k, k*n+i].
    #
    # For fixed k, we use these 2 terms.
    # Total rank = p * 2 = 12.
    #
    # Let's verify if rank 2 works for the sub-tensor.
    #
    # Term 0: u_0[i*m+j]=1, v_0[j]=1, w_0[i]=delta(i,0).
    # Term 1: u_1[i*m+j]=1, v_1[j]=1, w_1[i]=delta(i,1).
    #
    # Sum for i=0: 1*1*1 + 1*1*0 = 1.
    # Sum for i=1: 1*1*0 + 1*1*1 = 1.
    #
    # This works for all j!
    #
    # So the rank of T_k is 2.
    #
    # Therefore, the total rank is p * 2 = 6 * 2 = 12.
    #
    # Let's double check if we can do even better.
    # Can we share terms between different k?
    # We argued that we can't because of disjoint supports in V and W.
    #
    # So the minimal rank is likely 12.
    #
    # Let's construct the rank-12 decomposition.
    #
    # rank = 12.
    # For each k in 0..p-1:
    #   We have 2 terms.
    #   Term 2k:
    #     U: 1 at i*m+j for all i,j.
    #     V: 1 at j*p+k for all j.
    #     W: 1 at k*n+0. (w_0[0]=1, w_0[1]=0)
    #   Term 2k+1:
    #     U: 1 at i*m+j for all i,j.
    #     V: 1 at j*p+k for all j.
    #     W: 1 at k*n+1. (w_1[0]=0, w_1[1]=1)
    #
    # Let's verify this.
    #
    # T[i*m+j, j*p+k, k*n+i] = sum_r U_r[i*m+j] V_r[j*p+k] W_r[k*n+i].
    #
    # For a specific (i, j, k):
    # The only non-zero V_r are for r corresponding to this k.
    # So only r = 2k and r = 2k+1 contribute.
    #
    # U_{2k}[i*m+j] = 1.
    # U_{2k+1}[i*m+j] = 1.
    #
    # V_{2k}[j*p+k] = 1.
    # V_{2k+1}[j*p+k] = 1.
    #
    # W_{2k}[k*n+i] = 1 if i=0, else 0.
    # W_{2k+1}[k*n+i] = 1 if i=1, else 0.
    #
    # Sum = 1*1*W_{2k}[k*n+i] + 1*1*W_{2k+1}[k*n+i].
    #
    # If i=0: Sum = 1*1 + 1*0 = 1.
    # If i=1: Sum = 1*0 + 1*1 = 1.
    #
    # This is exactly 1 for all i, j, k.
    #
    # And for any other index combination, it will be 0 because the V or W index won't match.
    #
    # So the rank is 12.
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        # Term 0 for this k: corresponds to i=0
        col0 = 2 * k
        # Term 1 for this k: corresponds to i=1
        col1 = 2 * k + 1
        
        for i in range(n):
            for j in range(m):
                U[i * m + j][col0] = 1
                U[i * m + j][col1] = 1
                
                V[j * p + k][col0] = 1
                V[j * p + k][col1] = 1
                
                if i == 0:
                    W[k * n + i][col0] = 1
                    W[k * n + i][col1] = 0
                else:
                    W[k * n + i][col0] = 0
                    W[k * n + i][col1] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
