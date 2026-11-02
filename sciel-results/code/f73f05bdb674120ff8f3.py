import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 118.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # For matrix multiplication, the tensor is T[i,j,k] = delta (identity in the sum over j)
    # Actually for matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For standard matmul, T[i,j,k] = 1 if we're "selecting" that term
    # Wait, the tensor for matmul has T[i,j,k] = 1 for all valid i,j,k
    # and the decomposition must reproduce the bilinear map
    
    # Actually, the matmul tensor is: for each (i,k) output, we need sum_j A[i,j]*B[j,k]
    # The rank-1 decomposition term is U[i,j,r]*V[j,k,r]*W[k,i,r]
    # For the standard multiplication: T[i,j,k] = 1 (the tensor has all ones)
    # and we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # This is the matrix multiplication tensor of dimensions n x m x p
    # The standard rank is n*m*p (trivial)
    # Strassen gives lower rank for 2x2x2
    
    # Let me think about this more carefully.
    # The matmul tensor T of shape (n, m, p) has T[i,j,k] = 1 for all i,j,k
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    
    # For the trivial decomposition, rank = n*m*p, with U[i*m+j, r]=1 for r=(i,j), V[j*p+k,r]=1 for r=(j,k), W[k*n+i,r]=1 for r=(k,i)
    # But that doesn't work because r needs to be the same index...
    
    # Actually for matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # Rank-1 term: U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For trivial: each term (i,j,k) contributes 1
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for r = (i,j,k)
    # This gives T[i,j,k] = 1. Rank = n*m*p = 60.
    
    # For Strassen 2x2x2: 7 terms instead of 8
    
    # Let me build a systematic approach:
    # 1. Start with the trivial 60-rank decomposition
    # 2. Try to apply Strassen-like merges on 2x2x2 sub-blocks
    # 3. Try to further reduce
    
    # Strategy: Use a greedy/iterative approach to find minimal rank
    # Since n=2, m=5, p=6, we can try various block decompositions
    
    # For 2x5x6: 
    # We can decompose as:
    # - 2x2x6 part: use Strassen on the 2x2 block, 6 k-values -> 7*3 + ... 
    #   Actually 2x2x2 Strassen gives 7 for each 2x2x2 block
    #   2x2x6 = 2x2x2 * 3 blocks? No, it's one 2x2x6 tensor
    #   2x2x6 can be done with rank: we need to compute C[2,6] = A[2,2]*B[2,6]
    #   Using Strassen on the 2x2 part: 7 multiplications for each... 
    #   Actually 2x2xn matmul: rank is at most 7*ceil(n/2) + ... 
    #   For n=6: 2x2x6. We can use 3 pairs of Strassen: 7*3=21? No...
    #   2x2xn: A is 2x2, B is 2xn. C is 2xn.
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is just 4 multiplications per k, or we can use Strassen on the 2x2 part
    #   for each pair of k values.
    #   2x2x2: rank 7 (Strassen)
    #   2x2x6: we can group k into 3 pairs: 3 * 7 = 21? 
    #   Or: 2x2x6 = 2x2x2 + 2x2x2 + 2x2x2, each rank 7, total 21
    #   But we can do better: 2x2xn has rank at most 2n+... 
    #   Actually for 2x2xn: rank is 2n for n>=2? No, Strassen gives 7 for n=2.
    #   For 2x2xn: the tensor has 2*2*n = 4n entries.
    #   Rank lower bound: max(2*2, 2*n, 2*n) = max(4, 2n, 2n)
    #   For n=6: lower bound 12. Upper bound: 3*7=21 or maybe less.
    
    # Let me just try a systematic search with the given dimensions.
    
    # Approach: Generate candidate rank-1 tensors and use greedy matching
    # with rank reduction via linear algebra.
    
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Flatten: T_flat of size n*m*p, all ones
    
    # A rank-1 term (u, v, w) where u in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    # contributes to T_flat: u[i*m+j] * v[j*p+k] * w[k*n+i]
    # This is a specific pattern in the n*m*p dimensional space
    
    # For the trivial decomposition:
    # For each (i,j,k): u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i)
    # This is n*m*p = 60 terms
    
    # Let me implement the known optimal or near-optimal decompositions:
    # 
    # For 2x5x6:
    # Known results: 
    # - 2x2x2: rank 7
    # - 2x2xn: various
    # - 2x3xn: various
    # 
    # Let me try a constructive approach:
    # Split the 5 columns of A and 5 rows of B into blocks
    # A is 2x5, B is 5x6, C is 2x6
    # 
    # Option 1: Split into 2x2 + 2x2 + 1x1 blocks in the j dimension
    #   j=0,1: 2x2x6 subproblem
    #   j=2,3: 2x2x6 subproblem  
    #   j=4: 2x1x6 subproblem
    #
    # 2x2x6: A[2,2] * B[2,6]
    #   Using 3 Strassen 2x2x2 blocks: rank 21
    #   Or better: 2x2x6 can be done with rank... 
    #   Let me think: C[i,k] = sum_{j=0,1} A[i,j]*B[j,k]
    #   For each k: 2 mults. Total 12 mults trivially.
    #   Strassen on 2x2x2: 7 mults per 2 output entries (in k)
    #   For 6 k-values: 3 pairs -> 3*7 = 21? That's worse than 12.
    #   So for 2x2x6, trivial 12 is better than Strassen 21.
    #
    #   Actually wait: 2x2x2 Strassen computes 4 outputs with 7 mults.
    #   2x2x6 computes 12 outputs. With Strassen: 3*7=21 mults for 12 outputs.
    #   Trivial: 12 mults for 12 outputs. So trivial is better here.
    #
    # Option 2: For 2x2xn where n is small, use direct multiplication
    #   2x2x2: rank 7 (better than 8)
    #   2x2x3: rank? 
    #   2x2x4: rank?
    #
    # Let me just enumerate small cases:
    # 2x2x2: rank 7
    # 2x2x3: at most 7+2=9 (one Strassen + two direct) or 2x2x2 + 2x2x1
    #   2x2x1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0], rank 2
    #   So 2x2x3: rank 7+2=9
    # 2x2x4: 7+7=14 or 2x2x2+2x2x2=14, or maybe 12?
    #   Actually 2x2x4: C[2,4]=A[2,2]*B[2,4]
    #   Trivial: 8 mults. Strassen: 14. So trivial 8 is better.
    #
    # Hmm, Strassen only helps for 2x2x2 (7<8) and larger blocks where
    # the overhead is amortized.
    #
    # For 2x2xn: trivial rank = 4n/2 = 2n... wait.
    # 2x2xn: A is 2x2, B is 2xn, C is 2xn
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Each C[i,k] needs 2 mults, total 4n mults? No, 2*2*n = 4n terms.
    # But rank: each term (i,j,k) is one rank-1, so trivial rank = 4n? 
    # No wait: for fixed (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is 2 bilinear terms, so 2 rank-1 tensors.
    # Total: 2 * (number of (i,k) pairs) = 2 * 2 * n = 4n rank-1 terms.
    # But we can share: for fixed j, A[i,j]*B[j,k] for all i,k is one "block"
    # A[:,j] (2x1) * B[j,:] (1xn) = outer product, which is rank 1 in the tensor sense?
    # No, it's not rank 1 in the matmul tensor.
    #
    # Let me be precise. The matmul tensor T_{i,j,k} for 2x2xn:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,n-1}
    # Rank-1 decomposition: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    # For the trivial: r = (i,j,k), u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i)
    # This gives n*2*2 = 4n terms.
    #
    # For 2x2x2: 8 trivial, 7 Strassen.
    # For 2x2x3: 12 trivial. Can we do better?
    #   Maybe 7 + 2 = 9 (one Strassen block + 2 direct for the 3rd k)?
    #   Or 7 + 3 = 10?
    #   Let me try: use Strassen for k=0,1 and direct for k=2.
    #   Strassen for k=0,1: 7 terms covering (i,j,k) for k in {0,1}
    #   Direct for k=2: 4 terms (i=0,1; j=0,1; k=2)
    #   Total: 7+4 = 11? But Strassen covers all i,j for k=0,1.
    #   So 7 (for k=0,1) + 4 (for k=2) = 11.
    #   Can we do 10? 9?
    #
    # This is getting complex. Let me just use a computational approach.
    
    # I'll implement a greedy algorithm with rank-1 tensor candidates
    # and try to minimize the rank.
    
    # Generate all "simple" rank-1 candidates:
    # Type 1: single (i,j,k) - the trivial ones
    # Type 2: Strassen-like patterns on 2x2x2 blocks
    # Type 3: other structured patterns
    
    # Actually, let me try a different approach:
    # Since the problem is small (2x5x6), let me try to find the optimal
    # by searching over block decompositions.
    
    # The key insight: for 2x2x2, Strassen gives rank 7.
    # For other sizes, we may need to combine.
    
    # Let me try:
    # Split j into blocks of size 2 and 1:
    # j = [0,1], [2,3], [4]
    # For each block of size 2: 2x2x6 subproblem
    # For the block of size 1: 2x1x6 subproblem
    
    # 2x1x6: A[2,1]*B[1,6] = outer product, rank 6 (one per k)
    #   Actually: C[i,k] = A[i,0]*B[0,k] for i=0,1 and k=0..5
    #   That's 12 terms trivially, but rank 6:
    #   For each k: u[i,0]=1 for i=0,1, v[0,k]=1, w[k,i]=1
    #   Wait: u is (i,j), v is (j,k), w is (k,i)
    #   For j=0 fixed: u[i,0]*v[0,k]*w[k,i]
    #   If u[i,0]=1, v[0,k]=1, w[k,i]=1 for all i,k: that's 6 terms (one per k)
    #   Each term: u has 2 entries (i=0,1), v has 1 entry (k), w has 2 entries (i=0,1)
    #   So 6 rank-1 terms for 2x1x6.
    
    # 2x2x6: Let me figure out the optimal rank.
    # Lower bound: max(2*2, 2*6, 2*6) = max(4,12,12) = 12
    # Trivial: 12
    # Can we do better than 12? Probably not for 2x2x6.
    # So 2x2x6 has rank 12.
    
    # Total: 12 + 12 + 6 = 30
    # That's better than 60!
    
    # Can we do even better?
    # What about 2x3x6?
    # 2x3x6: A[2,3]*B[3,6]
    # Lower bound: max(6, 18, 12) = 18
    # Trivial: 36
    # Can we do 18?
    # 
    # Or 2x4x6?
    # Lower bound: max(8, 24, 12) = 24
    # Trivial: 48
    
    # The minimum of the sum of lower bounds over partitions of {0..4} into blocks:
    # Block size s: 2xsx6, lower bound max(2s, 12, 12) = max(2s, 12)
    # s=1: 12
    # s=2: 12
    # s=3: 12
    # s=4: 16
    # s=5: 20
    # 
    # So for any block of size >= 3, lower bound is 12.
    # For block of size 1 or 2, lower bound is 12.
    # Wait, for s=1: max(2,12,12)=12
    # For s=2: max(4,12,12)=12
    # For s=3: max(6,12,12)=12
    # For s=4: max(8,12,12)=12
    # For s=5: max(10,12,12)=12
    # 
    # So the lower bound for any 2xsx6 is 12!
    # That means 2x5x6 has lower bound 12?
    # 
    # But that can't be right. The trivial rank is 60.
    # The lower bound max(nm, mp, np) = max(10, 12, 12) = 12.
    # 
    # Can we achieve rank 12 for 2x5x6?
    # That would be amazing. Let me think...
    # 
    # Actually, the lower bound is max(nm, mp, np) where n=2,m=5,p=6
    # nm=10, mp=30, np=12. So lower bound = 30.
    # 
    # Wait, I need to be more careful.
    # The tensor is of size n x m x p = 2 x 5 x 6.
    # The rank is at least max(nm, mp, np) = max(10, 30, 12) = 30.
    # 
    # So the lower bound is 30. Can we achieve 30?
    # 
    # For 2x5x6 with rank 30:
    # We need U[10, 30], V[30, 30], W[12, 30]
    # 
    # This is essentially the "minimum rank" question.
    # For matrix multiplication, the minimum rank is known as the
    # bilinear complexity.
    # 
    # For 2x5x6, I'm not sure of the exact minimum, but 30 is the lower bound.
    # 
    # Let me try to construct a rank-30 solution.
    # 
    # Idea: For each k (0..5), we have a 2x5 subproblem:
    # C[:,k] = A * B[:,k] (2x5 times 5x1 = 2x1)
    # This is a 2x5x1 tensor. Rank of 2x5x1:
    # Lower bound: max(10, 5, 2) = 10
    # Trivial: 10
    # Can we do better? For 2x5x1: C[i] = sum_j A[i,j]*B[j,k]
    # This is just 10 multiplications, and each is a separate rank-1 term.
    # Actually no: for fixed k, C[i,k] = sum_j A[i,j] B[j,k]
    # Each term A[i,j]*B[j,k] is rank 1 in the tensor.
    # 10 terms for one k. For 6 k's: 60 terms.
    # But we can share across k's!
    # 
    # For fixed (i,j): A[i,j] * B[j,k] for all k is a "row" of B[j,:].
    # This gives: u[i,j]=1, v[j,k]=1 for all k, w[k,i]=1 for all k
    # That's not rank 1 because w has 6 entries and v has 6 entries.
    # 
    # Hmm, let me think differently.
    # 
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # 
    # One approach: use the fact that 2x5x6 can be decomposed as:
    # 2x2x6 + 2x2x6 + 2x1x6
    # Each 2x2x6: rank 12 (lower bound = max(4,12,12)=12)
    # Each 2x1x6: rank 6 (lower bound = max(2,6,6)=6)
    # Total: 12+12+6 = 30
    # 
    # This matches the lower bound! So rank 30 is achievable.
    # 
    # Now I need to construct the actual rank-1 terms.
    
    # For 2x2x6 (i in {0,1}, j in {j0,j0+1}, k in {0..5}):
    # T[i,j,k] = 1
    # Rank 12: For each k: 2x2x1 subproblem
    # 2x2x1: C[i,k] = A[i,j0]*B[j0,k] + A[i,j0+1]*B[j0+1,k]
    # 4 terms: (i=0,j=j0), (i=0,j=j0+1), (i=1,j=j0), (i=1,j=j0+1)
    # Each term: u[i,j]=1, v[j,k]=1, w[k,i]=1
    # So for 2x2x6: 4*6 = 24 terms? But lower bound is 12.
    # 
    # Wait, I think I'm confusing myself. Let me re-examine.
    # 
    # The tensor T[i,j,k] has dimensions n x m x p.
    # A rank-1 decomposition: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    # 
    # For the trivial decomposition of 2x2x6:
    # For each (i,j,k): u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i)
    # That's 2*2*6 = 24 terms.
    # 
    # But the lower bound is max(nm, mp, np) = max(4, 12, 12) = 12.
    # 
    # Can we do 12? Let me try:
    # For each k: we need to compute C[0,k] and C[1,k]
    # C[0,k] = A[0,j0]*B[j0,k] + A[0,j0+1]*B[j0+1,k]
    # C[1,k] = A[1,j0]*B[j0,k] + A[1,j0+1]*B[j0+1,k]
    # 
    # For fixed k: 4 multiplications.
    # Can we do it in 2? That would be like Strassen for 2x2x1.
    # 2x2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    #          C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is just 4 bilinear terms. Rank 4? Or can we do 3?
    # Actually 2x2x1 is just a 2D outer product problem.
    # C = A @ B where A is 2x2, B is 2x1.
    # Rank of this tensor: 4 (each entry is independent).
    # 
    # So 2x2x1 has rank 4, and 2x2x6 has rank 24? But lower bound is 12.
    # 
    # I think the issue is that the lower bound max(nm,mp,np) is for the
    # tensor rank in a different sense. Let me look at this more carefully.
    # 
    # The matrix multiplication tensor has a specific structure.
    # The rank (bilinear complexity) of 2x2x2 matmul is 7 (Strassen).
    # The rank of 2x2x1 matmul is 4 (can't be improved).
    # 
    # For 2x2xn: the rank is 4n for n=1, and for n>=2 it's at most 7*ceil(n/2)+4*floor(n/2)... 
    # Actually, for 2x2xn:
    # - n=1: rank 4
    # - n=2: rank 7 (Strassen)
    # - n=3: rank 7+4=11 or maybe 10?
    # - n=4: rank 7+7=14 or 12?
    # 
    # Hmm, this is getting complicated. Let me just use a computational approach.
    
    # I'll implement a greedy algorithm that tries to find a low-rank decomposition.
    
    n, m, p = DIMENSIONS
    target_size = n * m * p  # 60
    
    # Target tensor: all ones
    T = np.ones(target_size)
    
    # Generate candidate rank-1 tensors
    # Each candidate is a triple (u, v, w) where:
    # u: sparse vector in R^{n*m}
    # v: sparse vector in R^{m*p}
    # w: sparse vector in R^{p*n}
    # The contribution to T is: u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    candidates = []
    
    # Type 1: Standard unit tensors (trivial)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m+j] = 1.0
                v[j*p+k] = 1.0
                w[k*n+i] = 1.0
                # Compute the tensor entry pattern
                tensor = np.zeros(target_size)
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            if u[ii*m+jj] != 0 and v[jj*p+kk] != 0 and w[kk*n+ii] != 0:
                                tensor[ii*m*p + jj*p + kk] = u[ii*m+jj] * v[jj*p+kk] * w[kk*n+ii]
                candidates.append((u, v, w, tensor))
    
    # Type 2: Strassen 2x2x2 blocks
    def strassen_block(i0, j0, k0):
        """Generate 7 Strassen rank-1 terms for 2x2x2 block starting at (i0,j0,k0)"""
        terms = []
        # M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
        tensor = np.zeros(target_size)
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                    if val != 0:
                        tensor[ii*m*p+jj*p+kk] = val
        terms.append((u, v, w, tensor))
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1])*B[j0,k0]
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+k0] = 1
        w[k0*n+(i0+1)] = 1; w[(k0+1)*n+(i0+1)] = -1
        tensor = np.zeros(target_size)
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                    if val != 0:
                        tensor[ii*m*p+jj*p+kk] = val
        terms.append((u, v, w, tensor))
        
        # M3 = A[i0,j0]*(B[j0,k0+1]-B[j0+1,k0+1])
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1
        v[j0*p+(k0+1)] = 1; v[(j0+1)*p+(k0+1)] = -1
        w[(k0+1)*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
        tensor = np.zeros(target_size)
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                    if val != 0:
                        tensor[ii*m*p+jj*p+kk] = val
        terms.append((u, v, w, tensor))
        
        # M4 = A[i0+1,j0+1]*(B[j0+1,k0]-B[j0,k0])
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m+(j0+1)] = 1
        v[(j0+1)*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+(i0+1)] = 1
        tensor = np.zeros(target_size)
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                    if val != 0:
                        tensor[ii*m*p+jj*p+kk] = val
        terms.append((u, v, w, tensor))
        
        # M5 = (A[i0,j0]+A[i0,j0+1])*B[j0+1,k0+1]
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i0*m+(j0+1)] = 1
        v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = -1; w[(k0+1)*n+i0] = 1
        tensor = np.zeros(target_size)
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                    if val != 0:
                        tensor[ii*m*p+jj*p+kk] = val
        terms.append((u, v, w, tensor))
        
        # M6 = (A[i0+1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k0+1])
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+(k0+1)] = 1
        w[(k0+1)*n+(i0+1)] = 1
        tensor = np.zeros(target_size)
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                    if val != 0:
                        tensor[ii*m*p+jj*p+kk] = val
        terms.append((u, v, w, tensor))
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1])*(B[j0+1,k0]+B[j0+1,k0+1])
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+(j0+1)] = 1; u[(i0+1)*m+(j0+1)] = -1
        v[(j0+1)*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = 1
        tensor = np.zeros(target_size)
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                    if val != 0:
                        tensor[ii*m*p+jj*p+kk] = val
        terms.append((u, v, w, tensor))
        
        return terms
    
    # Add Strassen blocks for all possible 2x2x2 positions
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                for term in strassen_block(i0, j0, k0):
                    candidates.append(term)
    
    # Type 3: 2x2x1 patterns (for handling odd-sized blocks)
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k in range(p):
                # Various 2x2x1 patterns
                # Pattern: u has 2 entries in i, v has 1 entry, w has 2 entries in i
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[i0*m+j0] = 1; u[(i0+1)*m+j0] = 1
                v[j0*p+k] = 1
                w[k*n+i0] = 1; w[k*n+(i0+1)] = 1
                tensor = np.zeros(target_size)
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                            if val != 0:
                                tensor[ii*m*p+jj*p+kk] = val
                candidates.append((u, v, w, tensor))
    
    # Type 4: 1x2x2 patterns
    for i in range(n):
        for j0 in range(m-1):
            for k0 in range(p-1):
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[i*m+j0] = 1; u[i*m+(j0+1)] = 1
                v[j0*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
                w[k0*n+i] = 1; w[(k0+1)*n+i] = 1
                tensor = np.zeros(target_size)
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            val = u[ii*m+jj]*v[jj*p+kk]*w[kk*n+ii]
                            if val != 0:
                                tensor[ii*m*p+jj*p+kk] = val
                candidates.append((u, v, w, tensor))
    
    # Precompute tensor vectors for fast dot products
    num_cands = len(candidates)
    cand_tensors = np.array([c[3] for c in candidates])  # (num_cands, 60)
    cand_norms_sq = np.sum(cand_tensors**2, axis=1)
    cand_norms_sq_safe = np.where(cand_norms_sq > 1e-15, cand_norms_sq, 1.0)
    
    # Greedy selection
    residual = T.copy()
    selected = []
    
    for _ in range(100):
        if time.time() - start_time > DEADLINE - 10:
            break
        dots = cand_tensors @ residual
        reductions = dots**2 / cand_norms_sq_safe
        best = np.argmax(reductions)
        if reductions[best] < 1e-10:
            break
        coeff = dots[best] / cand_norms_sq_safe[best]
        selected.append((best, coeff))
        residual -= coeff * cand_tensors[best]
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Try to reduce rank
    if len(selected) > 1:
        sel_tensors = np.array([cand_tensors[s[0]] for s in selected])
        sel_coeffs = np.array([s[1] for s in selected])
        
        for _pass in range(20):
            if time.time() - start_time > DEADLINE - 15:
                break
            if len(selected) <= 1:
                break
            improved = False
            for idx in range(len(selected)):
                if len(selected) - 1 < 1:
                    break
                remaining = [i for i in range(len(selected)) if i != idx]
                rem_tensors = np.array([cand_tensors[sel_tensors[i,0]] for i in remaining])
                try:
                    x = np.linalg.lstsq(rem_tensors.T, T, rcond=None)[0]
                    recon = rem_tensors @ x
                    if np.linalg.norm(recon - T) < 1e-8:
                        selected = [(sel_tensors[remaining[i],0], x[i]) for i in range(len(remaining))]
                        improved = True
                        break
                except:
                    pass
            if not improved:
                break
    
    # Build the result
    if len(selected) > 0 and np.linalg.norm(residual) < 1e-8:
        rank = len(selected)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for r, (cand_idx, coeff) in enumerate(selected):
            u, v, w, tensor = candidates[cand_idx]
            U[:, r] = u * coeff
            V[:, r] = v
            W[:, r] = w
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m))
        B_test = np.random.randint(-5, 5, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_calc[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: trivial decomposition (rank 60)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

