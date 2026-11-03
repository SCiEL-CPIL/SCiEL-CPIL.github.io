"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Try to find a low-rank decomposition
    # Use a structured approach based on blocking
    
    # For 2x4x6, try to use the fact that we can decompose into blocks
    # Let's try rank 24 first (half of 48)
    
    # Strategy: Use the middle dimension (size 4) and split into two groups of 2
    # For each pair of middle indices, use a 2x2x6 sub-problem
    # Actually, let's try a direct construction
    
    # Try: use 2x2 Strassen-like blocks
    # Split j into j1 (0,1) and j2 (2,3)
    # Split k into k1 (0,1), k2 (2,3), k3 (4,5)
    
    # For each (j_group, k_group) pair, we have a 2x2x2 sub-tensor
    # There are 2 j_groups and 3 k_groups, so 6 sub-tensors of size 2x2x2
    # Each 2x2x2 can be done with rank 7 (Strassen)
    # Total rank: 6 * 7 = 42? That's not better than 48 much.
    
    # Better idea: share structure across the k_groups
    # Let's try a different approach: use rank 32
    
    # Actually, let me try to construct a valid decomposition with rank r
    # and verify it, searching for the minimum r
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    # Try to find a decomposition with decreasing rank
    # Start with a known valid one (schoolbook) and try to reduce
    
    # Let's try a structured construction for rank 32
    # Use the fact that T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Idea: use a "sum" structure
    # Let U[i*m+j, c] = a[i,j,c], V[j*p+k, c] = b[j,k,c], W[k*n+i, c] = d[k,i,c]
    # sum_c a[i,j,c] * b[j,k,c] * d[k,i,c] = 1
    
    # For this to hold for all i,j,k, one simple solution is:
    # If we set a[i,j,c] = 1/sqrt(rank), b[j,k,c] = 1/sqrt(rank), d[k,i,c] = 1/sqrt(rank)
    # then sum_c = rank * (1/rank)^(3/2) = 1/sqrt(rank), not 1.
    
    # Better: a[i,j,c] = 1, b[j,k,c] = 1, d[k,i,c] = 1/rank
    # then sum_c = rank * 1 * 1 * (1/rank) = 1. This works! But it's rank-dependent.
    
    # So a trivial rank-r decomposition:
    # U[i*m+j, c] = 1 for all i,j,c
    # V[j*p+k, c] = 1 for all j,k,c  
    # W[k*n+i, c] = 1/rank for all k,i,c
    # This gives sum_c 1*1*(1/rank) = 1. 
    
    # But wait, this uses rank r with entries 1/rank. The task says "integer/half-integer entries"
    # 1/rank is not integer or half-integer unless rank divides 2.
    
    # So I need integer or half-integer entries.
    
    # Let me use: U = 2, V = 1, W = 1/(2*rank) ... no, still not good.
    
    # Try: U = 1, V = 1, W = 1/rank. For W to be half-integer, rank must be 1 or 2.
    
    # Alternative: use different values.
    # U[i*m+j, c] = 2, V[j*p+k, c] = 1, W[k*n+i, c] = 1/(2*rank)
    # Still problematic.
    
    # Let me use: U = 1, V = 2, W = 1/(2*rank). No.
    
    # How about: for each c, U=1, V=1, W=1/rank. Need 1/rank in {0, 1/2, 1, 3/2, ...}
    # So rank must be 1 or 2 for W to be half-integer. That doesn't help.
    
    # Different approach: use a mix of values.
    # Let me try rank 4 with:
    # U all 1s, V all 1s, W: some entries 1/2, some 1/2, etc.
    # sum_c W[k*n+i, c] = 1, with each W entry in {0, 1/2, 1, ...}
    # For rank 4: four 1/2's sum to 2, not 1. Three 1/2's and one -1/2? Sum = 1. 
    # But negative values might be OK? The task says "integer/half-integer", not non-negative.
    
    # Let me try: rank 4, U all 1s, V all 1s, W[k*n+i, 0]=1/2, W[k*n+i, 1]=1/2, W[k*n+i, 2]=1/2, W[k*n+i, 3]=-1/2
    # Sum = 1/2+1/2+1/2-1/2 = 1. 
    # But this requires U and V to be all 1s, which means the decomposition doesn't actually depend on i,j,k properly.
    # Wait, T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # If U=1, V=1 for all, then T[i,j,k] = sum_c W[k*n+i,c] = 1 for all i,j,k. That's correct!
    
    # So rank 4 works with U=1s, V=1s, W with entries summing to 1 per (k,i) pair.
    # But is this a valid tensor decomposition? The rank of the tensor should be the minimum r such that
    # T can be written as sum of r rank-1 tensors. A rank-1 tensor is u_i * v_j * w_k.
    # In our case, each term is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
    # For this to be a rank-1 tensor in the (i,j,k) space, we need U[i*m+j,c] = a_i * b_j, 
    # V[j*p+k,c] = b_j * c_k, W[k*n+i,c] = c_k * a_i (or similar factorization).
    # But in general, U[i*m+j,c] is just an arbitrary value, not necessarily factorable.
    
    # Wait, I think I'm confusing myself. The decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a valid representation of the tensor as a sum of "rank-1" terms in a specific sense,
    # where each term is the outer product of three vectors (one in the (i,j) space, one in (j,k) space, one in (k,i) space).
    # The rank is the number of such terms.
    
    # So my construction with U=1s, V=1s, W summing to 1 is valid! And it has rank 4.
    # But that seems too good. Let me double-check.
    
    # T[i,j,k] = sum_c 1 * 1 * W[k*n+i, c] = sum_c W[k*n+i, c]
    # For this to equal 1 for all i,j,k, we need sum_c W[k*n+i, c] = 1 for all k,i.
    # With rank 4: W[k*n+i, 0]=1/2, W[k*n+i, 1]=1/2, W[k*n+i, 2]=1/2, W[k*n+i, 3]=-1/2
    # Sum = 1. Yes!
    
    # But wait, is this really a valid matrix multiplication decomposition? The issue is that
    # in matrix multiplication, the tensor T[i,j,k] represents the contribution of A[i,j]*B[j,k] to C[i,k].
    # The decomposition should compute C[i,k] = sum_j A[i,j]*B[j,k] using rank-many multiplications.
    # My construction above doesn't actually compute the matrix product correctly because it ignores the 
    # structure of how A and B are multiplied.
    
    # I think the correct interpretation is:
    # We have matrices A (n x m) and B (m x p), and we want C = A * B (n x p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 represents that each (i,j,k) triple contributes A[i,j]*B[j,k] to C[i,k].
    # A rank-r decomposition means we can compute all C[i,k] using r scalar multiplications of linear forms in A and B.
    # Specifically: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # Wait, that's not quite right either.
    
    # Let me re-read the problem. The tensor decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 for all i,j,k.
    # This is a mathematical identity on the tensor, not directly about computing the matrix product.
    # The "rank" is the number of terms in this decomposition.
    
    # So my construction is valid as a tensor decomposition! The rank is 4.
    # But that seems impossibly low. Let me verify with a small case.
    
    # For 1x1x1: T[0,0,0] = 1. Rank 1: U[0,0]=1, V[0,0]=1, W[0,0]=1. Sum = 1. Correct.
    # For 2x2x2: T[i,j,k] = 1 for all i,j,k in {0,1}. 
    # My construction: U all 1s (size 4x4), V all 1s (size 4x4), W with sum 1 per (k,i).
    # T[i,j,k] = sum_c W[k*2+i, c] = 1. Correct!
    # But the known rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen), not 4.
    
    # So something is wrong with my understanding. Let me reconsider.
    
    # Ah, I think the issue is that in a proper tensor decomposition for matrix multiplication,
    # the rank-1 terms must correspond to actual rank-1 tensors in the tensor product space.
    # The tensor T is in the space R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} (or similar).
    # A rank-1 tensor in this space is u ⊗ v ⊗ w where u ∈ R^{n*m}, v ∈ R^{m*p}, w ∈ R^{p*n}.
    # So T = sum_c u_c ⊗ v_c ⊗ w_c, and the rank is the minimum r.
    # In coordinates: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But T is only defined for valid (i,j,k) triples, and T[i,j,k] = 1.
    # The full tensor has entries for all (a,b,c) where a ∈ [0,n*m), b ∈ [0,m*p), c ∈ [0,p*n),
    # but T[a,b,c] is only non-zero when a = i*m+j, b = j'*p+k, c = k'*n+i' with j = j' and k = k'.
    # Wait, no. The tensor T for matrix multiplication has T[i,j,k] = 1 for all valid i,j,k,
    # and it's zero otherwise. But in the flattened representation, T[a,b,c] is non-zero only when
    # the indices are consistent (the j index in a matches the j' in b, etc.).
    
    # I think the correct interpretation is that T is a tensor in R^n ⊗ R^m ⊗ R^p, with T[i,j,k] = 1.
    # And the decomposition is T = sum_c u_c ⊗ v_c ⊗ w_c where u_c ∈ R^n, v_c ∈ R^m, w_c ∈ R^p.
    # But the problem statement says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # So U[i*m+j, c] is the (i,j) component of the c-th vector in R^{n*m}.
    # This means the decomposition is in the space R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}, not R^n ⊗ R^m ⊗ R^p.
    
    # So T is a tensor in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}, with T[a,b,c] = 1 when a = i*m+j, b = j*p+k, c = k*n+i
    # for some valid i,j,k, and 0 otherwise.
    
    # For n=2, m=4, p=6:
    # T[a,b,c] = 1 if there exist i,j,k with i ∈ {0,1}, j ∈ {0,1,2,3}, k ∈ {0,1,2,3,4,5}
    # such that a = i*4+j, b = j*6+k, c = k*2+i.
    # Otherwise T[a,b,c] = 0.
    
    # So my earlier construction was wrong because I was assuming T[i,j,k] = 1 for all i,j,k,
    # but actually T is sparse in the flattened representation.
    
    # Let me reconstruct: T[a,b,c] = 1 if a mod 4 = b // 6 and (b mod 6) * 2 + (a // 4) = c.
    # i.e., j = a % m, i = a // m, k = b % p, j2 = b // p, i2 = c % n, k2 = c // n
    # T[a,b,c] = 1 if j == j2 and k == k2 and i == i2.
    
    # So T[a,b,c] = 1 iff (a % m) == (b // p) and (b % p) == (c // n) and (a // m) == (c % n).
    
    # Now, my construction with U=1s, V=1s, W summing to 1 would give:
    # sum_c U[a,c] * V[b,c] * W[c_index,c] = sum_c 1 * 1 * W[c_index,c] = 1 for all a,b,c.
    # But T[a,b,c] is 0 for most (a,b,c), so this doesn't work.
    
    # I need a decomposition where sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c] = 1 or 0.
    
    # This is much harder. Let me go back to the schoolbook decomposition and try to optimize.
    
    # Schoolbook: for each (i,j,k), one term with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # Rank = n*m*p = 48.
    
    # To reduce rank, I need to find linear dependencies or use a smarter construction.
    
    # Let me try to use the fact that for fixed i,k, T[i*m+j, j*p+k, k*n+i] = 1 for all j.
    # So for fixed i,k, the vector (T[i*m+0, 0*p+k, k*n+i], T[i*m+1, 1*p+k, k*n+i], ..., T[i*m+3, 3*p+k, k*n+i]) = (1,1,1,1).
    
    # This suggests that for fixed i,k, I can use a single "j-sum" term.
    
    # Idea: for each (i,k), use a term that sums over j.
    # U[i*m+j, c] = 1 for all j, V[j*p+k, c] = 1 for all j, W[k*n+i, c] = 1/4.
    # Then sum over j: but the decomposition is sum_c, not sum_j.
    
    # I think I need to accept that the schoolbook rank of 48 might be close to optimal,
    # and try to find a small improvement.
    
    # Let me try rank 47 by finding one linear dependency.
    # Or try a different approach: use a known lower bound.
    
    # The rank of the n×m×p matrix multiplication tensor is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
    # So rank ≥ 24.
    
    # Let me try to construct a rank-24 decomposition.
    # For this, I need sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c].
    
    # One approach: use the fact that T can be written as a sum of rank-1 tensors.
    # For example, T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}.
    # This is the schoolbook decomposition with rank 48.
    
    # To reduce rank, I can try to combine terms.
    # For example, if I have two terms with the same U and V but different W, I can combine them.
    
    # Let me try a programmatic search for a low-rank decomposition.
    
    # Given the time constraint, let me just return the schoolbook decomposition with rank 48.
    # But let me try one more idea: use the "laser method" or a known result for rectangular matrices.
    
    # For 2×4×6, I'll try to use a block decomposition.
    # Split the 4×6 matrix into 2×(2×3) blocks.
    # Then the multiplication is like (2×2) * (2×3) + (2×2) * (2×3) for the two row blocks of B.
    
    # Actually, let me just try to construct a decomposition with rank 32 and verify it.
    
    # Use the following construction:
    # For each i in {0,1}, k in {0,1,2,3,4,5}, j in {0,1,2,3}:
    # Instead of one term per (i,j,k), try to share terms.
    
    # Idea: for fixed i, use a 4×6 sub-problem (m×p matrix multiplication with n=1).
    # The tensor for n=1, m=4, p=6 has T[j,k] = 1 for all j,k.
    # Rank of this tensor: it's a 4×6 matrix of 1s, which has rank 1 as a matrix,
    # but as a tensor in R^4 ⊗ R^6, the rank is... well, T[j,k] = 1 = sum_c u[j,c]*v[k,c].
    # With u[j,c] = 1, v[k,c] = 1, c=0..0 (rank 1): sum = 1. So rank 1 for n=1.
    
    # But for n=2, we have T[i,j,k] = 1, and the decomposition is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    # This is different from the n=1 case.
    
    # I think I'll just go with the schoolbook decomposition for now, as finding a better one
    # within the time constraint is challenging.
    
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
