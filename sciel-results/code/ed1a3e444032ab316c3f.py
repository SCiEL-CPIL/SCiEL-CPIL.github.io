"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check: for each (i, k, a, b, c, d), 
    # sum_r W_arr[r, k*n+i] * U_arr[r, a*m+b] * V_arr[r, c*p+d] 
    # should equal 1 if a==i and b==c and d==k, else 0.
    
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            s = np.sum(W_arr[:, k*n+i] * U_arr[:, a*m+b] * V_arr[:, c*p+d])
                            if abs(s - expected) > tol:
                                return False
    return True

def _make_schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _try_pair_combine(n, m, p):
    """Try to combine pairs of schoolbook terms to reduce rank by 1.
    
    Schoolbook term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be combined into one term
    if there exist scalars a,b such that:
    a*U1 + b*U2 has support in {i1*m+j1, i2*m+j2} (or subset)
    a*V1 + b*V2 has support in {j1*p+k1, j2*p+k2}
    a*W1 + b*W2 has support in {k1*n+i1, k2*n+i2}
    And the combined term correctly reproduces both original contributions.
    
    Actually, let's think more carefully. The schoolbook tensor is:
    T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    We want to find a decomposition of rank < n*m*p.
    One approach: if two terms share a factor, we might combine them.
    """
    # For 2x5x5, let's try a specific structured approach.
    # The key insight: for fixed j, the terms (0,j,k) and (1,j,k) differ only in the U and W indices.
    # U[0*m+j] vs U[1*m+j] - different positions
    # W[k*n+0] vs W[k*n+1] - different positions
    # V[j*p+k] - same position!
    
    # So for fixed (j,k), we have two terms that share the V factor.
    # Term1: U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
    # Term2: U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
    
    # Can we replace these 2 terms with 1 term?
    # We need: a*e_{j} + b*e_{m+j} (in U space, positions j and m+j)
    #           a*e_{j*p+k} + b*e_{j*p+k} = (a+b)*e_{j*p+k} (in V space)
    #           a*e_{k*n} + b*e_{k*n+1} (in W space, positions k*n and k*n+1)
    
    # For this to equal the sum of the two original terms, we'd need:
    # a*e_j ⊗ (a+b)*e_{j*p+k} ⊗ a*e_{kn} + ... this doesn't work simply.
    
    # Let's try a different approach: use the structure of small matrix multiplication.
    # For 2x5 * 5x5, the tensor has rank at most 50 (schoolbook).
    # Known results: for n=2, the rank of 2xm * mp is at most m*p*2 - something.
    
    # Let's try: for each pair of columns in the "middle" dimension,
    # see if we can save a term.
    
    n, m, p = 2, 5, 5
    rank = n * m * p  # 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Start with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to reduce rank by combining terms.
    # For each pair (j1, j2) in the middle dimension, try to see if terms
    # (0,j1,k) and (1,j2,k) or similar can be merged.
    
    # Actually, let's try a more systematic approach:
    # Try to find a rank-49 decomposition by solving a linear system.
    
    # The matrix multiplication tensor T has entries:
    # T[a,b,c,d] = 1 if a==i, b==c, d==k for some i,k, else 0
    # More precisely: T is a 4-tensor where T[i,j,j',k] = delta_{i,i} * delta_{j,j'} * delta_{k,k}
    # Wait, let me re-index. A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,k,j] = ... no, it's a 3-way tensor for bilinear maps.
    # T[i,j,k,l] where (i,j) indexes A, (k,l) indexes B, and the output is (i',l')
    # Actually for the decomposition:
    # sum_r U[a,b,r] * V[c,d,r] * W[i,k,r] = delta_{a,i} * delta_{b,c} * delta_{d,k}
    
    # Let's try to find a rank-49 solution by setting up the problem.
    # We'll fix 49 of the 50 schoolbook terms and see if the 50th can be absorbed.
    
    # For each term to potentially eliminate:
    for elim_i in range(n):
        for elim_j in range(m):
            for elim_k in range(p):
                # Try to eliminate term (elim_i, elim_j, elim_k)
                # Keep all other 49 terms, and see if we can adjust them
                # to also cover the eliminated term.
                
                # This is a linear algebra problem.
                # The 49 kept terms give us 49 vectors in the tensor space.
                # We need to check if the eliminated term is in their span.
                
                # The tensor space has dimension n*m * m*p * p*n = 10*25*10 = 2500
                # Each term is a rank-1 tensor.
                
                # Let's build the matrix of the 49 kept terms and check if
                # the eliminated term is in their span.
                
                kept_terms = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i, j, k) == (elim_i, elim_j, elim_k):
                                continue
                            kept_terms.append((i, j, k))
                
                # Build matrix: each column is a rank-1 tensor flattened
                # Tensor for (i,j,k): U has 1 at (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
                # Flattened tensor: e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
                # This is a vector of size (n*m)*(m*p)*(p*n) = 10*25*10 = 2500
                # with a single 1 at position ((i*m+j)*(m*p) + (j*p+k))*(p*n) + (k*n+i)
                
                # Actually, the tensor is T[a,b,c,d] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For schoolbook term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
                # So the tensor entry T[i*m+j, j*p+k, k*n+i] = 1, all others 0.
                
                # We need to check if the vector e_{(elim_i*m+elim_j, elim_j*p+elim_k, elim_k*n+elim_i)}
                # is in the span of the 49 kept basis vectors.
                
                # Since all these are standard basis vectors, the eliminated term is in the span
                # iff it equals one of the kept terms, which it doesn't.
                # So we can't simply eliminate a term.
                
                # We need to allow the kept terms to have modified coefficients.
                # This becomes: find scalars s_1, ..., s_49 such that
                # sum_t s_t * e_{(i_t*m+j_t, j_t*p+k_t, k_t*n+i_t)} = e_{(elim_i*m+elim_j, elim_j*p+elim_k, elim_k*n+elim_i)}
                # This only works if the eliminated term equals one of the kept terms, which it doesn't.
                
                # So we need a different approach: allow the decomposition to use
                # non-standard basis vectors.
                
                pass
    
    return None  # No improvement found

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try structured approaches to reduce rank
    # For 2x5 x 5x5, let's try to exploit the small n=2 dimension.
    
    # Key idea: For fixed j (column of A / row of B), we have terms:
    # (0,j,k) and (1,j,k) for k=0..4
    # These share V[j*p+k] but have different U and W.
    
    # Another idea: use the fact that C[0,k] and C[1,k] are related.
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # Let's try a specific construction:
    # For each j, define S_j = A[0,j] + A[1,j] and D_j = A[0,j] - A[1,j]
    # Then A[0,j] = (S_j + D_j)/2, A[1,j] = (S_j - D_j)/2
    # C[0,k] = sum_j (S_j + D_j)/2 * B[j,k] = 1/2 * sum_j S_j*B[j,k] + 1/2 * sum_j D_j*B[j,k]
    # C[1,k] = sum_j (S_j - D_j)/2 * B[j,k] = 1/2 * sum_j S_j*B[j,k] - 1/2 * sum_j D_j*B[j,k]
    
    # So: C[0,k] = (X_k + Y_k)/2, C[1,k] = (X_k - Y_k)/2
    # where X_k = sum_j S_j * B[j,k], Y_k = sum_j D_j * B[j,k]
    
    # X_k = sum_j (A[0,j]+A[1,j]) * B[j,k]
    # Y_k = sum_j (A[0,j]-A[1,j]) * B[j,k]
    
    # Now, X_k is a dot product of the vector S=(S_0,...,S_4) with row k of B.
    # Similarly for Y_k.
    
    # The rank-1 decomposition for X_k:
    # X_k = sum_j S_j * B[j,k]
    # U for S_j: U[j] = 1 (for both A[0,j] and A[1,j] contributing)
    # Actually, S_j = A[0,j] + A[1,j], so U[j, r] should be 1 for both row 0 and row 1?
    # No, U has size n*m = 10. U[0*m+j] corresponds to A[0,j], U[1*m+j] to A[1,j].
    # For S_j = A[0,j] + A[1,j]: U[0*m+j] = 1, U[1*m+j] = 1
    # For D_j = A[0,j] - A[1,j]: U[0*m+j] = 1, U[1*m+j] = -1
    
    # So for each j and k, we have:
    # Term for S_j * B[j,k]: U[0*m+j]=1, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=1/2, W[k*n+1]=1/2
    # Term for D_j * B[j,k]: U[0*m+j]=1, U[1*m+j]=-1, V[j*p+k]=1, W[k*n+0]=1/2, W[k*n+1]=-1/2
    
    # This gives us 2*m*p = 50 terms, same as schoolbook. No improvement.
    
    # But wait, can we combine terms across different j values?
    # X_k = sum_j S_j * B[j,k] is already a sum of m terms.
    # The total number of terms is 2*m*p = 50.
    
    # Let's try a different splitting. What if we group by k instead?
    # For each k, C[0,k] = sum_j A[0,j]*B[j,k] and C[1,k] = sum_j A[1,j]*B[j,k]
    # These are 2*p independent dot products of length m.
    # Each dot product of length m has rank m (in general).
    # Total: 2*p*m = 50.
    
    # The key question: is there a way to share factors between different (i,k) pairs?
    
    # Let's try: for each j, consider the vector (A[0,j], A[1,j]) and the row (B[j,0], ..., B[j,p-1]).
    # The contribution of index j to C is: outer((A[0,j], A[1,j]), (B[j,0],...,B[j,p-1]))
    # = A[0,j]*B[j,0] at (0,0), A[0,j]*B[j,1] at (0,1), ..., A[1,j]*B[j,p-1] at (1,p-1)
    
    # This is a rank-1 matrix in the n x p space, scaled by the "interaction" at index j.
    # The total C is sum_j of these rank-1 matrices.
    # The rank of C as a matrix is at most min(n,p) = 2, but that's not helpful for tensor rank.
    
    # For tensor rank, we need to decompose the bilinear map.
    # The bilinear map is: (A,B) -> sum_j outer(A[:,j], B[j,:])
    
    # This is a sum of m=5 terms, each of which is a rank-1 bilinear map.
    # So the tensor rank is at most 5? No, that's not right.
    # Each term outer(A[:,j], B[j,:]) is itself a bilinear map that requires
    # n*p = 10 scalar multiplications in general (for n=2, p=5).
    # So total: 5 * 10 = 50.
    
    # Unless we can share factors between different j values.
    # For term j: U has nonzeros at positions (0,j) and (1,j) [i.e., 0*m+j and 1*m+j]
    # V has nonzeros at positions (j,0), (j,1), ..., (j,p-1) [i.e., j*p+0, ..., j*p+p-1]
    # W has nonzeros at all positions (0,0), (0,1), ..., (1,p-1) [i.e., 0*n+0, ..., 1*n+p-1]
    
    # For different j values, the U and V supports are disjoint, but W support is the same.
    # So we can't easily combine terms from different j.
    
    # Let's try a completely different approach: use known results for small matrix multiplication.
    # For 2xm * mp, the rank is at most 2mp - something.
    # Known: rank(2x2 * 2x2) = 7 (Strassen)
    # rank(2x3 * 3x2) = ? 
    # rank(2x5 * 5x5) = ?
    
    # For 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    # The saving comes from: (A+B)(C+D) - A*C - B*D gives one product "for free" in some sense.
    
    # For 2x5 * 5x5, can we apply a similar idea?
    # Let's try to apply Strassen-like tricks to pairs of columns.
    
    # Consider columns j and j+1 of A, and the corresponding rows of B.
    # Let A1 = [A[0,j], A[1,j]] and A2 = [A[0,j+1], A[1,j+1]]
    # Let B1 = [B[j,0], ..., B[j,p-1]] and B2 = [B[j+1,0], ..., B[j+1,p-1]]
    
    # Contribution to C from these two indices:
    # outer(A1, B1) + outer(A2, B2)
    # = [[A1[0]*B1[0], A1[0]*B1[1], ..., A1[0]*B1[p-1]],
    #    [A1[1]*B1[0], A1[1]*B1[1], ..., A1[1]*B1[p-1]]]
    # + [[A2[0]*B2[0], A2[0]*B2[1], ..., A2[0]*B2[p-1]],
    #    [A2[1]*B2[0], A2[1]*B2[1], ..., A2[1]*B2[p-1]]]
    
    # Can we compute this with fewer than 2*p = 10 scalar multiplications?
    # This is like multiplying two 2x1 matrices by two 1x5 matrices.
    # The total "size" is 2*1*5 = 10 scalar mults in schoolbook.
    
    # Using Strassen-like: (A1+A2)(B1+B2) - A1*B1 - A2*B2 = cross terms
    # But we need A1*B1 + A2*B2, not the cross terms.
    
    # Actually, for 2x1 * 1x5, the rank is 5 (can't do better in general).
    # For two such products, the rank is at most 10, but maybe we can do 9?
    
    # Let's try: define S = A1 + A2, D = A1 - A2
    # S*B1 = A1*B1 + A2*B1
    # S*B2 = A1*B2 + A2*B2
    # D*B1 = A1*B1 - A2*B1
    # D*B2 = A1*B2 - A2*B2
    
    # A1*B1 + A2*B2 = (S*B1 + S*B2)/2 + (D*B1 - D*B2)/2
    # This uses 4 "big" multiplications (each of which is a 2x1 * 1x5 product = 5 scalar mults)
    # Total: 4*5 = 20 scalar mults, worse than 10.
    
    # So this doesn't help.
    
    # Let's try yet another approach: use the fact that p=5 is odd.
    # Maybe we can group 5 columns in a way that saves a term.
    
    # For now, let's just return the schoolbook solution.
    # The random search is extremely unlikely to find an exact decomposition,
    # and the structured approaches above haven't yielded an improvement.
    
    # Let's do one more thing: try to verify if there's a rank-49 solution
    # by solving a linear system.
    
    if still_going():
        # Try rank 49: fix 49 schoolbook terms and see if we can adjust
        # coefficients to also cover the 50th term.
        # This is equivalent to asking if the 50th standard basis vector
        # is in the span of the other 49, which it's not.
        # So we need to allow non-standard basis vectors.
        
        # Let's try: take the schoolbook 50 terms, and try to perturb
        # one of them to also cover another term.
        
        # For example, take term (0,0,0) and try to modify it to also
        # contribute to (0,0,1).
        # Original: U[0]=1, V[0]=1, W[0]=1 (for (0,0,0))
        # We want: U[0]=1, V[0]=1, W[0]=1 AND also contribute to (0,0,1)
        # For (0,0,1): need U[0]=1, V[1]=1, W[1]=1
        # So we'd need V to have 1s at both position 0 and 1.
        # Modified term: U[0]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=1
        # But this would also contribute to (0,1,0) and (0,1,1) incorrectly.
        
        # So we need to cancel out the unwanted contributions.
        # This gets complicated. Let's just return schoolbook.
        
        pass
    
    return best
# EVOLVE-BLOCK-END
