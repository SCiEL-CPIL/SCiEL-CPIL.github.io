import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
DEADLINE = 145.0  # safety margin before 150s

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Naive rank = n*m*p = 96
    naive_rank = n * m * p
    
    # Build naive decomposition
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    # Verify the naive solution
    U_np = np.array(U_naive, dtype=float)
    V_np = np.array(V_naive, dtype=float)
    W_np = np.array(W_naive, dtype=float)
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Check time
    if time.time() - START_TIME > DEADLINE:
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    # Try to find a lower-rank decomposition using the Winograd/block approach
    # Strategy: Use the fact that we can decompose the problem into blocks
    # 3x4 * 4x8 can be split as:
    #   Row blocks of A: [2x4] and [1x4]
    #   Column blocks of B: [4x4] and [4x4]
    # 
    # So we have 4 sub-problems:
    #   (2x4)*(4x4) -> 2x4 result
    #   (2x4)*(4x4) -> 2x4 result  
    #   (1x4)*(4x4) -> 1x4 result
    #   (1x4)*(4x4) -> 1x4 result
    #
    # For each 2x4*4x4 sub-problem, we can further split:
    #   (2x2)*(2x4) + (2x2)*(2x4)
    #   Each 2x2*2x4 can use Strassen-like approach
    #
    # Actually, let me think about this differently.
    # The tensor rank of matrix multiplication M_{n,m,p} is at most n*m*p (naive)
    # but can be much lower.
    #
    # For 2x2x2, the rank is 7 (Strassen)
    # For 2x2x4, we can compute as 2x2 * 2x2 twice, so rank <= 7+7 = 14, but actually
    # we can do better by sharing the Strassen terms.
    #
    # Let me try a direct approach: use known fast algorithms for small blocks
    # and combine them.
    
    # Approach: Use the "block" decomposition where we split m=4 into two blocks of 2
    # A is 3x4, B is 4x8
    # Split A into A1 (3x2) and A2 (3x2)
    # Split B into B1 (2x8) and B2 (2x8)
    # C = A1*B1 + A2*B2
    #
    # Each A_i * B_i is a 3x2 * 2x8 multiplication
    # For 3x2 * 2x8: split 3 into 2+1, 8 into 4+4
    #   (2x2)*(2x4) + (2x2)*(2x4) + (1x2)*(2x4) + (1x2)*(2x4)
    #
    # For 2x2 * 2x4: this is like multiplying a 2x2 by a 2x4
    # We can view this as two 2x2*2x2 multiplications, but they share the left factor.
    # Actually, 2x2 * 2x4 has tensor rank at most 7+7=14, but we can do better.
    #
    # Let me try a different approach: directly construct a rank-88 decomposition
    # by finding 8 terms that can be eliminated from the naive 96.
    
    # Alternative: Use the fact that for the (i,j,k) -> 1 tensor, we can use
    # the identity that allows us to combine certain terms.
    
    # Let me try: for the 2x2x2 block, use Strassen (rank 7 instead of 8)
    # We have blocks:
    # i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    #
    # Split i into {0,1} and {2}
    # Split j into {0,1} and {2,3}
    # Split k into {0,1,2,3} and {4,5,6,7}
    #
    # This gives 2x2x4 blocks for the (0,1)x(0,1)x(0,1,2,3) part, etc.
    
    # Let me try a more systematic approach using the "embedding" technique.
    # We know that M_{2,2,2} has rank 7.
    # M_{3,4,8} can be embedded in a larger structure.
    
    # Actually, let me just try to find any improvement over 96 by using
    # the Strassen decomposition on 2x2x2 sub-blocks.
    
    # Split the problem into 8 sub-problems of size 2x2x2 (using i={0,1},{2} -> pad to 2x2x2)
    # Actually this is getting complicated. Let me try a direct algebraic approach.
    
    # Key insight: The tensor T[i,j,k] = 1 for all i,j,k can be written as:
    # T[i,j,k] = (sum_a u_a[i] v_a[j]) * (sum_b u'_b[j] v'_b[k]) * (sum_c u''_c[k] v''_c[i])
    # No, that's not right. The tensor is rank-1 in each mode separately but 
    # the full tensor rank is what we're minimizing.
    
    # Let me try: use the fact that T[i,j,k] = 1 = (1)*(1)*(1) and look for
    # low-rank decompositions using the "border rank" or "approximate rank" ideas,
    # but we need exact.
    
    # For now, let me try to implement a rank-88 decomposition by finding
    # 8 cancellations in the naive 96-term decomposition.
    
    # The naive decomposition has terms e_{i,j,k} for all (i,j,k).
    # We want to find 88 rank-1 tensors whose sum is sum_{i,j,k} e_{i,j,k}.
    
    # One approach: use the identity
    # e_{i,j,k} = (1/2) * (e_{i,j,k} + e_{i',j',k'}) for some pairing
    # This doesn't directly help.
    
    # Let me try a different strategy: use the "Kronecker product" structure.
    # 3x4*4x8 can be seen as a Kronecker-like product of smaller problems.
    
    # 3 = 2+1, 4 = 2+2, 8 = 2+2+2+2
    # This doesn't factor nicely.
    
    # Let me just try to use the Strassen algorithm on 2x2x2 blocks and see
    # if we can get a valid decomposition with rank < 96.
    
    # Build a decomposition using 2x2x2 Strassen blocks:
    # Split i into groups of 2 (with padding), j into groups of 2, k into groups of 2
    # i: {0,1}, {2} -> pad to {2,?} 
    # This is messy. Let me try a cleaner approach.
    
    # Direct approach: Try to find a rank-88 decomposition by solving a system of equations.
    # This is too expensive for 88 variables.
    
    # Let me try the simplest improvement: use the fact that for the last index k,
    # we have 8 values, and for j we have 4, and for i we have 3.
    
    # Actually, let me just return the naive solution since finding a provably
    # lower-rank exact decomposition for 3x4x8 is non-trivial.
    
    # But wait - let me try one more thing: use the "Winograd" approach for the
    # 2x2 * 2x2 sub-blocks.
    
    # Split A (3x4) into:
    #   A_top = A[0:2, :] (2x4)
    #   A_bot = A[2:3, :] (1x4)
    # Split B (4x8) into:
    #   B_left = B[:, 0:4] (4x4)
    #   B_right = B[:, 4:8] (4x4)
    
    # C = A_top * B_left + A_top * B_right + A_bot * B_left + A_bot * B_right
    #   = [A_top * B_left, A_top * B_right]
    #     [A_bot * B_left, A_bot * B_right]
    
    # For A_top * B_left (2x4 * 4x4):
    #   Split A_top into A_top1 = A_top[:, 0:2] (2x2) and A_top2 = A_top[:, 2:4] (2x2)
    #   Split B_left into B_left1 = B_left[0:2, :] (2x4) and B_left2 = B_left[2:4, :] (2x4)
    #   A_top * B_left = A_top1 * B_left1 + A_top2 * B_left2
    #   Each term is 2x2 * 2x4
    
    # For 2x2 * 2x4: 
    #   This can be done with rank at most 7 (using a generalized Strassen)
    #   Actually, 2x2 * 2x4 has tensor rank at most 7+7 = 14 (two 2x2*2x2),
    #   but we can share computations.
    #   
    #   Let me think: C = A * B where A is 2x2, B is 2x4.
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #   C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    #
    #   The tensor T[i,j,k] for this is 1 for all i in {0,1}, j in {0,1}, k in {0,1,2,3}
    #   So it's the same as 2x2x4 tensor of all 1s.
    #   
    #   Rank of 2x2x4 all-ones tensor: at most 7 (since 2x2x2 has rank 7, and we can
    #   replicate for the 4th dimension... actually no, that gives 14).
    #   
    #   Actually, the 2x2x4 all-ones tensor has rank at most 7 because we can use:
    #   The same 7 Strassen terms work for each pair of k values, but we need to
    #   handle the k dimension.
    #   
    #   Let me just use rank 14 for each 2x2*2x4 (two independent 2x2*2x2).
    
    # Total rank estimate:
    # 4 sub-problems of type 2x4*4x4, each split into 2 terms of 2x2*2x4
    # 8 sub-problems of type 2x2*2x4, each with rank 14
    # Plus 4 sub-problems of type 1x4*4x4, each split into 2 terms of 1x2*2x4
    # 8 sub-problems of type 1x2*2x4, each with rank 8 (naive) or less
    #
    # Total: 8*14 + 8*8 = 112 + 64 = 176. That's worse than 96!
    
    # So the block approach with naive sub-solutions is worse. We need to use
    # fast algorithms for the sub-problems.
    
    # For 2x2*2x4, the rank is actually 7 (not 14)! Here's why:
    # We can use the 7 Strassen terms and apply them to both halves of B.
    # The U and V matrices are the same for both halves, only W changes.
    # So we get 7 terms for the first 2 columns of B and 7 for the second 2,
    # but the U and V are shared, so total rank is 7+7=14 for the full 2x2*2x4.
    # Wait, that's still 14.
    
    # Hmm, actually for 2x2*2x4, the tensor is:
    # T[i,j,k] = 1 for i,j in {0,1}, k in {0,1,2,3}
    # This is the "all-ones" tensor of size 2x2x4.
    # Its rank is at most 7 (since it contains the 2x2x2 all-ones tensor as a sub-tensor,
    # and the rank is at least 7). Actually, the rank of the 2x2x4 all-ones tensor
    # is exactly 7! Because we can write it as the sum of 7 rank-1 tensors using
    # the Strassen decomposition extended to the 4th dimension.
    
    # Let me verify: the Strassen 7 terms for 2x2x2 give us T[i,j,k]=1 for k in {0,1}.
    # For k in {2,3}, we need the same T[i,j,k]=1.
    # We can use the same U and V terms, and new W terms for k=2,3.
    # But that gives 7+7=14 terms total.
    
    # Unless... we can combine. The W matrix for the Strassen terms has entries
    # for k=0 and k=1. For k=2 and k=3, we need the same pattern.
    # So we need 7 terms for (k=0,1) and 7 for (k=2,3), total 14.
    
    # So 2x2*2x4 has rank 14. My earlier calculation was correct.
    
    # This means the block approach gives rank 176, which is worse than 96.
    # So we should stick with the naive 96.
    
    # Let me try one more idea: use the "border rank" or look for a specific
    # algebraic identity that reduces the rank by a small amount.
    
    # For the 3x4x8 all-ones tensor, can we find 8 terms that cancel?
    # The tensor is T = sum_{i=0}^{2} sum_{j=0}^{3} sum_{k=0}^{7} e_{i,j,k}
    # where e_{i,j,k} is the rank-1 tensor with 1 at (i,j,k) and 0 elsewhere.
    
    # We want to write T as a sum of 88 rank-1 tensors.
    # Each rank-1 tensor is u * v * w where u is a vector of length 3,
    # v is a vector of length 4, w is a vector of length 8.
    
    # The sum of 88 such tensors should equal the all-ones tensor.
    
    # This is a system of 96 equations (one for each (i,j,k)) in 88*3 + 88*4 + 88*8 = 88*15 = 1320 unknowns.
    # So there should be many solutions. But finding one is hard.
    
    # Let me try a greedy approach: start with 96 naive terms and try to merge
    # pairs of terms into single terms.
    
    # Two terms e_{i1,j1,k1} + e_{i2,j2,k2} can be merged if there exist
    # u, v, w such that u*v*w = e_{i1,j1,k1} + e_{i2,j2,k2}.
    # This requires that u has support on {i1,i2}, v on {j1,j2}, w on {k1,k2},
    # and the product matches. This is only possible if the two terms share
    # the same i, j, or k index in a specific way.
    
    # Actually, e_{i1,j1,k1} + e_{i2,j2,k2} is a rank-2 tensor (in the 3-tensor sense)
    # unless i1=i2, j1=j2, or k1=k2 in a way that allows factorization.
    
    # If i1=i2, j1=j2, k1!=k2: then e_{i,j,k1} + e_{i,j,k2} = (e_i * e_j) * (e_{k1} + e_{k2})
    # which is rank-1! So we can merge two terms that differ only in k.
    # Similarly for i and j.
    
    # So we can merge all terms with the same (i,j) into one term with a w vector
    # that has 1s in all k positions. Similarly for (i,k) and (j,k).
    
    # Using (i,j) grouping: 3*4 = 12 terms, each with w having 8 ones.
    # Using (i,k) grouping: 3*8 = 24 terms, each with v having 4 ones.
    # Using (j,k) grouping: 4*8 = 32 terms, each with u having 3 ones.
    
    # The (i,j) grouping gives rank 12! That's much better than 96!
    
    # Let me verify: T[i,j,k] = sum_{r} u_r[i] * v_r[j] * w_r[k]
    # If we use 12 terms, one for each (i,j) pair:
    #   u_{i,j}[i'] = 1 if i'=i else 0
    #   v_{i,j}[j'] = 1 if j'=j else 0  
    #   w_{i,j}[k'] = 1 for all k'
    # Then sum_{i',j'} u_{i',j'}[i] * v_{i',j'}[j] * w_{i',j'}[k]
    # = sum_{i',j'} delta(i',i) * delta(j',j) * 1
    # = 1 (when i'=i and j'=j, which is exactly one term)
    
    # Wait, that gives T[i,j,k] = 1 for all (i,j,k). Yes!
    # So the rank is at most 12!
    
    # Let me double-check: the tensor is T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}.
    # Using 12 terms (one per (i,j)):
    #   Term (i,j): u has 1 at position i, v has 1 at position j, w has 1s everywhere.
    #   Product: u[i]*v[j]*w[k] = 1*1*1 = 1 when (i',j')=(i,j), else 0.
    #   Sum over all (i',j'): only the term with i'=i, j'=j contributes, giving 1.
    
    # So the rank is at most 12! That's a huge improvement over 96.
    
    # Can we do even better? The rank is at least max(n, m, p) = 8 (since the tensor
    # has "slice rank" at least max dimension). Actually, the rank is at least the
    # "slice rank" which is at least max(n, m, p) = 8.
    
    # Can we achieve rank 8? Let me think...
    # We need 8 rank-1 tensors whose sum is the all-ones tensor.
    # Each rank-1 tensor is u_r * v_r * w_r.
    # We need sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all (i,j,k).
    
    # This means that for each (i,j), sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all k.
    # So the vector (sum_r u_r[i] * v_r[j] * w_r[0], ..., sum_r u_r[i] * v_r[j] * w_r[7])
    # must be (1,1,...,1) for all (i,j).
    
    # This means that the matrix M[i,j] = sum_r u_r[i] * v_r[j] must satisfy
    # M[i,j] * w_r = ... hmm, this is getting complicated.
    
    # Actually, let me just go with rank 12 for now. It's a huge improvement over 96.
    
    # Build the rank-12 decomposition:
    rank_12 = n * m  # 3 * 4 = 12
    
    U_12 = [[0] * rank_12 for _ in range(n * m)]
    V_12 = [[0] * rank_12 for _ in range(m * p)]
    W_12 = [[0] * rank_12 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # u has 1 at position i*m + j (but we only need u[i] = 1)
            # Actually, U is of shape (n*m, rank), so U[i*m + j, col] = 1
            U_12[i * m + j][col] = 1
            # v has 1 at all positions j*p + k for k in range(p)
            for k in range(p):
                V_12[j * p + k][col] = 1
            # w has 1 at all positions k*n + i for k in range(p)
            for k in range(p):
                W_12[k * n + i][col] = 1
            col += 1
    
    # Verify
    if verify(U_12, V_12, W_12, rank_12):
        return {"U": U_12, "V": V_12, "W": W_12}
    
    # Fallback to naive
    return {"U": U_naive, "V": V_naive, "W": W_naive}
