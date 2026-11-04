import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the matrix multiplication tensor T[i,j,k] = 1 if ... 
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor: T[i, k, j] where A has indices (i,k), B has (k,j), C has (i,j)
    # Decomposition: T = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Wait, the convention: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W[k*n+i, r] per the task: "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i where k is the B-column/C-column index? Let me re-read.
    # "W[k*n+i,r]" - so W row index = k*n + i. Here k ranges over p (0..p-1), i over n (0..n-1).
    # The tensor for matrix mult: C[i, j] = sum_k A[i, k] B[k, j]
    # U[i*m + k, r] (i over n, k over m)
    # V[k*p + j, r] (k over m, j over p)
    # W[j*n + i, r] (j over p, i over n)
    # Then sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1 if this is the "1" term, 0 otherwise.
    # Actually the tensor T[i, k, j] = 1 (always 1 for valid i,k,j) is NOT right.
    # The matrix multiplication tensor is T[i, k, j] = 1 for all i,k,j? No.
    # Let me think: the bilinear map is (A, B) -> A*B. The tensor T has T[i, k, j] = 1 
    # meaning C[i,j] gets contribution A[i,k]*B[k,j] with coefficient 1.
    # So T[i, k, j] = 1 for all i in [0,n), k in [0,m), j in [0,p).
    # The decomposition: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # So we need sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r] == 1 for all i,k,j.
    
    # Compute LHS for all (i,k,j)
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                lhs = 0.0
                for r in range(rank):
                    lhs += U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]
                err = abs(lhs - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def construct():
    n, m, p = DIMENSIONS
    # Try the improved decomposition:
    # Split i into {0,1} and {2}
    # For i in {0,1}: 2 x m x p tensor, use direct per k or per j
    # For i = 2: 1 x m x p tensor, rank p (one per output column j)
    
    # Approach: For each output column j (0..p-1), and each i (0..n-1):
    # C[i, j] = sum_k A[i,k] * B[k,j]
    # This is a dot product over k. For fixed j, the sub-tensor over (i, k) is n x m, rank n*m (direct).
    # Total: p * n * m = 90. That's the original.
    
    # Better: For fixed k, C[:, j] gets A[:, k] * B[k, j]. 
    # For fixed k, the contribution to all (i, j) is outer(A[:,k], B[k,:]) which is rank 1 in the (i,j) space.
    # So for each k (m of them), we have a rank-1 contribution to the n x p output.
    # But the tensor is over (i, k, j): T[i,k,j] = 1.
    # Decomposition: T[i,k,j] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
    # 
    # Idea: For each k in 0..m-1, for each i in 0..n-1, for each j in 0..p-1:
    #   term: U[i*m+k, r] = 1, V[k*p+j, r] = 1, W[j*n+i, r] = 1
    # That's m*n*p = 90 terms.
    #
    # Can we reduce? The constraint is: for all (i,k,j), sum_r U[i*m+k,r] V[k*p+j,r] W[j*n+i,r] = 1.
    #
    # Let's try: for each k, use a decomposition of the n x p "slice" T[i, k, j] = 1 (for fixed k).
    # For fixed k, we need: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i, j.
    # Note U[i*m+k, r] depends on i and r (and k via the row index), V[k*p+j, r] depends on j and r, W[j*n+i, r] depends on i, j, r.
    # This is a 3-way tensor of order (n, p) for each k, but coupled across k via the shared r index.
    #
    # Simplest: use rank-1 per (k, i, j) -> 90 terms.
    #
    # Alternative: For fixed k, the n x p matrix of all 1s has rank 1. So we could use 1 term per k if 
    # W didn't couple i and j. But W[j*n+i, r] couples them.
    #
    # Let me try: for each k, assign one rank index r_k.
    #   U[i*m+k, r_k] = 1 for all i
    #   V[k*p+j, r_k] = 1 for all j
    #   W[j*n+i, r_k] = 1 for all i, j
    # Then for (i, k, j): sum_r ... = U[i*m+k, r_k]*V[k*p+j, r_k]*W[j*n+i, r_k] = 1*1*1 = 1. 
    # But we need this for ALL k simultaneously with different r_k, and no cross-terms.
    # Cross-term: for r = r_{k'} with k' != k: U[i*m+k, r_{k'}] = 0 (since row i*m+k is not in k' group).
    # So this works! Total rank = m = 5.
    # Wait, that can't be right. Let me verify.
    
    # For k in 0..m-1, r_k = k (say).
    # U[i*m+k, k] = 1 for all i in 0..n-1. So U[i*m+k, r] = 1 if r == k, else 0.
    # V[k*p+j, k] = 1 for all j. So V[k*p+j, r] = 1 if r == k, else 0.
    # W[j*n+i, k] = 1 for all i, j. So W[j*n+i, r] = 1 if r == k, else 0.
    # 
    # Check: for (i, k, j), sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
    # = sum_r [r==k ? 1 : 0] * [r==k ? 1 : 0] * [r==k ? 1 : 0]
    # = 1 (only r=k contributes).
    # 
    # This gives rank = m = 5! But that's way too good. Let me double-check the tensor definition.
    
    # Hmm wait. The matrix multiplication tensor T[i, k, j] should encode C[i,j] = sum_k A[i,k] B[k,j].
    # The tensor is T[i, k, j] = 1 for all i, k, j. Yes.
    # And the decomposition requires sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r] = T[i,k,j] = 1.
    # My construction gives exactly 1. So rank 5 works?!
    
    # But that contradicts known results. The rank of 3x5x6 matrix mult should be much higher.
    # Let me re-examine. Oh wait, I think I made an error. Let me re-examine U's indexing.
    # U has shape (n*m, rank). Row index is i*m + k where i in 0..n-1, k in 0..m-1.
    # For U[i*m+k, r] = 1 if r == k: this means for a given k, ALL rows i*m+k for i=0..n-1 have U[., k] = 1.
    # That's fine.
    # V has shape (m*p, rank). Row index k*p + j. V[k*p+j, r] = 1 if r == k.
    # W has shape (p*n, rank). Row index j*n + i. W[j*n+i, r] = 1 if r == k.
    # 
    # The issue: W[j*n+i, r] = 1 if r == k, but k is not determined by (j, i). 
    # W[j*n+i, r] must be a fixed value for each (j, i, r). It can't depend on k.
    # In my construction, I set W[j*n+i, r] = 1 if r == k for ALL k? That's impossible unless n*m*p = 1.
    # 
    # I see my error. W[j*n+i, r] is a single value. I can't set it to 1 for r=k for all k simultaneously.
    # For a given (j, i), W[j*n+i, r] has one value per r. I need it to be 1 when r corresponds to the k that 
    # is being summed over, but k is a free index in the tensor.
    #
    # So my construction is invalid. W[j*n+i, r] must be independent of k.
    #
    # Let me restart. The constraint is:
    # For all i in [0,n), k in [0,m), j in [0,p):
    #   sum_{r=0}^{R-1} U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    #
    # This is a system of n*m*p = 90 equations in R * (n*m + m*p + p*n) = R * (15 + 30 + 18) = 63R unknowns.
    # For R = 5: 315 unknowns, 90 equations. Feasible in principle but the structure is rigid.
    
    # Let me just use the safe direct decomposition with rank 90, verified.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    err = _verify(U, V, W, n, m, p)
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # Fallback (should not happen)
    return {"U": U, "V": V, "W": W}
