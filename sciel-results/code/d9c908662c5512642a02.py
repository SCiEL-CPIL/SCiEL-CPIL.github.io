import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Block-diagonal embedding: 2x4x5 = (2x4x4) ⊕ (2x4x1)
    # 2x4x4: rank 24 (from 2x2x4 rank-12 tensor via Kronecker with I_2)
    # 2x4x1: rank 8 (schoolbook for 2x4x1)
    # Total rank: 32
    
    total_rank = 32
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    # 2x2x4 rank-12 decomposition patterns (indices: a[0..3], b[0..7], c[0..3])
    # These are the 12 rank-1 terms for multiplying 2x2 by 2x2 matrix with 4 columns
    # Using known rank-12 decomposition for 2x2x4
    rank12_patterns = [
        # (a_pattern, b_pattern, c_pattern) where a is 4-dim, b is 8-dim, c is 4-dim
        # a index: i*2+j, b index: j*4+k, c index: k*2+i
        # We need 12 patterns that sum to the identity tensor for 2x2x4
        
        # Let's construct from Strassen-like ideas for 2x2x2 blocks
        # 2x2x4 can be seen as two 2x2x2 blocks (columns 0-1 and 2-3)
        # Each 2x2x2 has rank 7, so naive would be 14, but we can share
        
        # Actually, let me use a different approach:
        # 2x2x4 tensor T[i,j,k] = 1 if j==j (always true in our case)
        # Wait, the tensor for matrix mult is T[i,j,k] where result[i,k] = sum_j A[i,j]*B[j,k]
        # For 2x2x4: i in 0..1, j in 0..1, k in 0..3
        # T[i, j, k] = 1 (the multiplication tensor has 1s at positions (i,j,k) for all valid i,j,k)
        # No wait - the multiplication tensor M[i,j,k] = delta_{j,j} which is just 1 for all (i,j,k)
        # That's not right either. Let me reconsider.
        
        # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] where:
        # result[i,k] = sum_j A[i,j] * B[j,k]
        # So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # The tensor is T = sum_r U_r ⊗ V_r ⊗ W_r where
        # U_r[i,j], V_r[j,k], W_r[k,i] are the factor vectors
        
        # For 2x2x2, rank 7 Strassen:
        # The 7 rank-1 terms have specific patterns.
        
        # For 2x2x4, we can view it as 2x2x2 ⊕ 2x2x2 but with shared j dimension.
        # Actually j goes 0..1, k goes 0..3.
        # T[i,j,k] = 1 for all i,j,k.
        # This is the outer product of ones vectors! So rank 1?
        # No - the constraint is that each rank-1 term is U_r[i,j] * V_r[j,k] * W_r[k,i]
        # and the sum over r must equal 1 for all (i,j,k).
        
        # Let me just hardcode a known rank-12 for 2x2x4.
        # Actually, I'll use a simpler approach: build 2x4x4 directly with rank 24.
    ]
    
    # Let me construct 2x4x4 directly using block structure
    # 2x4x4: i in 0..1, j in 0..3, k in 0..3
    # Split j into two blocks of 2: j1 = j//2 (0 or 1), j2 = j%2 (0 or 1)
    # Similarly k into two blocks: k1 = k//2, k2 = k%2
    
    # For each (j1, k1) pair, we have a 2x2x2 subproblem with indices (i, j2, k2)
    # But the j and k indices are shared across blocks in a specific way.
    
    # Actually, let's think of it differently:
    # 2x4x4 = sum over (j1,k1) of 2x2x2 tensors where:
    #   i is the 2-dim index
    #   j = j1*2 + j2 where j2 in 0..1
    #   k = k1*2 + k2 where k2 in 0..1
    
    # For each (j1, k1), the sub-tensor is M[i, j1*2+j2, k1*2+k2] = 1
    # This is a 2x2x2 tensor (in indices i, j2, k2) that equals 1 everywhere.
    
    # But these sub-tensors are not independent because the factor vectors must be consistent.
    
    # Let me use a Kronecker product approach:
    # 2x4x4 = (2x2x2) ⊗ (1x2x2) ? No, that doesn't work directly.
    
    # Alternative: 2x4x4 can be decomposed as:
    # For each of the 4 j-values and 4 k-values, we need coverage.
    # Rank 24 = 4 * 6, suggesting 4 blocks of rank 6 each.
    # Or 2 * 12, two blocks of rank 12.
    
    # Let me try: split k into two halves (k=0,1 and k=2,3)
    # For k=0,1: 2x4x2 subproblem
    # For k=2,3: 2x4x2 subproblem
    # Each 2x4x2 has rank at most 8 (schoolbook), so total 16. That's better than 24!
    
    # Wait, let me reconsider. 2x4x2: i in 0..1, j in 0..3, k in 0..1
    # T[i,j,k] = 1 for all i,j,k
    # Can we do better than rank 8?
    # 2x4x2 = (2x2x2) ⊕ (2x2x2) where we split j into j=0,1 and j=2,3
    # Each 2x2x2 has rank 7, so total rank 14? No, that's worse.
    # But we can share factors!
    
    # Actually for 2x4x2, the tensor is:
    # T[i, j, k] = 1
    # This can be written as: for each j, we need U[i,j] * V[j,k] * W[k,i] to sum to 1.
    # If we use U[i,j] = 1 for all, then we need sum_j V[j,k] * W[k,i] ... no that's not right.
    
    # Let me just use the known result: rank(2x4x4) <= 24 via specific construction.
    # I'll construct it as 4 copies of a 2x2x2 rank-6 subproblem... no.
    
    # Let me try a direct construction for 2x4x4 with rank 24:
    # Split into 4 blocks based on (j_block, k_block) where j_block = j//2, k_block = k//2
    # For each (jb, kb) in {0,1}x{0,1}, we have a 2x2x2 subproblem.
    # If each has rank 6, total is 24.
    
    # 2x2x2 rank 6 decomposition (one less than Strassen's 7):
    # This is not generally possible - the border rank might allow it but exact rank is 7.
    # So we need rank 7 for each, giving 28. Still better than schoolbook 32 for 2x4x4.
    
    # Hmm, let me reconsider the whole approach.
    
    # Schoolbook for 2x4x5: rank 40
    # Current best: rank 36
    # Target: rank 32 via 2x4x4 (rank 24) + 2x4x1 (rank 8)
    
    # For 2x4x4 with rank 24, I'll use the following construction:
    # View 2x4x4 as two 2x4x2 blocks (k=0,1 and k=2,3)
    # Each 2x4x2: use rank 12 construction
    # Total: 24
    
    # 2x4x2 rank 12 construction:
    # i in 0..1, j in 0..3, k in 0..1
    # Split j into two 2-blocks: j=0,1 and j=2,3
    # For j-block 0 (j=0,1): 2x2x2 subproblem, rank 7
    # For j-block 1 (j=2,3): 2x2x2 subproblem, rank 7
    # But we can potentially share some factors...
    # Actually no, the j indices are different so the factors are independent.
    # So 2x4x2 has rank at most 14 via two Strassen blocks.
    
    # That gives 2x4x4 rank at most 28, and 2x4x5 rank at most 28+8=36. Same as current!
    
    # I need a better construction. Let me think about 2x4x4 more carefully.
    
    # 2x4x4: T[i,j,k] = 1 for i in 0..1, j in 0..3, k in 0..3
    # = sum over r of U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Key insight: T[i,j,k] = 1 = 1 * 1 * 1, but we need the product structure.
    # Note that T[i,j,k] depends on i, j, k separately (it's constant).
    # So we need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    
    # If we set W_r[k,i] = 1 for all k,i (i.e., W_r is all-ones vector), then:
    # sum_r U_r[i,j] * V_r[j,k] = 1 for all i,j,k
    # This means the matrix (sum_r U_r[:,j] * V_r[j,:]) must be all-ones.
    # Let M_r = U_r[:,j] * V_r[j,:] (outer product for fixed j)... this is getting complex.
    
    # Let me try a different factorization.
    # Suppose we use rank-1 terms where:
    # U_r[i,j] = a_r(i) * b_r(j)
    # V_r[j,k] = c_r(j) * d_r(k)  
    # W_r[k,i] = e_r(k) * f_r(i)
    # Then the sum becomes sum_r a_r(i) b_r(j) c_r(j) d_r(k) e_r(k) f_r(i)
    # = sum_r [a_r(i) f_r(i)] * [b_r(j) c_r(j)] * [d_r(k) e_r(k)]
    # = sum_r A_r(i) * B_r(j) * C_r(k)
    # And we need this to equal 1 for all i,j,k.
    # This is a rank-1 decomposition of the all-ones tensor in the "separable" sense.
    # The all-ones tensor of size 2x4x4 has rank 1 in this separable sense!
    # Just take A_0(i)=1, B_0(j)=1, C_0(k)=1.
    # But wait, this corresponds to U_0[i,j]=1, V_0[j,k]=1, W_0[k,i]=1.
    # Check: sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1*1*1 = 1. Yes!
    # So 2x4x4 has rank 1?!
    
    # No wait, that can't be right. The issue is that in matrix multiplication,
    # the tensor is not T[i,j,k]=1. Let me re-examine.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor M has entries M[i,j,k] such that:
    # C[i,k] = sum_{i',j',k'} M[i',j',k'] * A[i',j'] * B[j',k'] * (delta_{i,i'} * delta_{k,k'})
    # Actually, the standard definition: the tensor T satisfies
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # where T[i,j,k] = 1 if the indices match the multiplication pattern.
    # Specifically, T[i,j,k] = 1 for all valid (i,j,k) because the multiplication
    # tensor is just the "identity" in the sense that each triple (i,j,k) contributes.
    
    # Hmm, but that would make it rank 1 as I showed above. That's clearly wrong.
    
    # Let me look at this more carefully. The multiplication tensor is defined such that:
    # <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] must satisfy: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k] for each (i,k)
    # This means T[i,j,k] = delta_{i,i} * delta_{k,k} ... no.
    
    # Actually, T is a 3-tensor where:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # The tensor representation is: C = T(A, B) where T is a multilinear map.
    # In terms of a 3D array, T[i,j,k] is the coefficient such that:
    # C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k']
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # T[i,j,k] = 1 if i'=i, j'=j, k'=k... no.
    
    # The correct statement: T is a tensor in (R^{n x m})^* ⊗ (R^{m x p})^* ⊗ R^{n x p}
    # Or equivalently, T has indices (i, j, k) where i indexes A's row, j indexes A's col/B's row,
    # k indexes B's col, and the output is (i, k).
    # So T[i, j, k] is the coefficient of A[i,j] * B[j,k] in the formula for C[i,k].
    # And indeed, C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all i,j,k
    # where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    # But then as I showed, this has rank 1! Which contradicts known results.
    
    # The resolution: the rank of a tensor is the minimum number of rank-1 tensors that sum to it.
    # A rank-1 tensor is u ⊗ v ⊗ w where (u ⊗ v ⊗ w)[i,j,k] = u[i] * v[j] * w[k].
    # But in our case, the factor vectors are:
    # U_r has shape (n*m,) indexed by (i,j)
    # V_r has shape (m*p,) indexed by (j,k)  
    # W_r has shape (p*n,) indexed by (k,i)
    # And the reconstruction is: T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    # So a rank-1 term has the form: U[i*m+j] * V[j*p+k] * W[k*n+i]
    # For this to equal 1 for all (i,j,k), we need:
    # U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all i,j,k
    # This requires U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1 for all indices.
    # And indeed, 1*1*1 = 1. So rank 1 works!
    
    # This means the multiplication tensor has rank 1, which is absurd.
    # I must be misunderstanding the tensor definition.
    
    # Let me re-read the task: "Find a rank-minimal exact tensor decomposition for multiplying
    # a 2 by 4 matrix by a 4 by 5 matrix."
    
    # The output factor W[k*n+i, r] suggests the indexing is different.
    # Let me re-examine: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # The reconstruction should give the multiplication tensor.
    
    # I think the issue is that the multiplication tensor is NOT the all-ones tensor.
    # Let me think again.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to write C[i,k] = sum_r (sum_j U_r[i,j] * A[i,j]) * (sum_k' V_r[j,k'] * B[j,k']) ... no.
    
    # Actually, the standard formulation:
    # C[i,k] = sum_r U_r[i,k] * (sum_j A[i,j] * V_r[j]) * (sum_j' B[j',k] * W_r[j'])
    # No, that's not right either.
    
    # The correct formulation for tensor decomposition of matrix multiplication:
    # C = A * B where C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_r (sum_j U_r[i,j] A[i,j]) * (sum_j' V_r[j',k] B[j',k]) ... no.
    
    # Let me look at it from the factor perspective:
    # U: (n*m) x rank, so U[i*m+j, r]
    # V: (m*p) x rank, so V[j*p+k, r]
    # W: (p*n) x rank, so W[k*n+i, r]
    
    # The reconstruction: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And T should represent the multiplication tensor such that:
    # C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k']
    
    # For C[i,k] = sum_j A[i,j] B[j,k], we need:
    # sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] = sum_j A[i,j] B[j,k]
    
    # This means T[i',j',k'] must be such that when contracted with A and B,
    # we get the correct result. By linearity, this requires:
    # T[i',j',k'] = delta_{i',i} * delta_{k',k} * ... no, T doesn't have i,k as free indices.
    
    # I think the correct interpretation is:
    # The multiplication tensor M is in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ R^{n*p}
    # Wait, W has shape (p*n, rank), so it's in R^{p*n}.
    # So M is in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ R^{p*n}
    
    # And M[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    
    # So the tensor entry at position (a, b, c) where a = i*m+j, b = j*p+k, c = k*n+i is 1.
    # Note that a, b, c are not independent - they're linked through i, j, k.
    # For a given (a, b, c), there may or may not exist (i, j, k) such that
    # a = i*m+j, b = j*p+k, c = k*n+i.
    
    # So the tensor is NOT all-ones. It's 1 only at positions (i*m+j, j*p+k, k*n+i)
    # for valid (i,j,k), and 0 elsewhere.
    
    # This makes more sense! The tensor is sparse in a structured way.
    
    # So for 2x4x5: n=2, m=4, p=5
    # a = i*4+j where i in 0..1, j in 0..3, so a in 0..7
    # b = j*5+k where j in 0..3, k in 0..4, so b in 0..19
    # c = k*2+i where k in 0..4, i in 0..1, so c in 0..9
    
    # T[a,b,c] = 1 if there exist i,j,k with a=i*4+j, b=j*5+k, c=k*2+i
    # Otherwise T[a,b,c] = 0.
    
    # This is the correct tensor! And it's not rank 1.
    
    # OK so my earlier "verify_decomposition_fast" was wrong. Let me fix it.
    # The target tensor should have T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k),
    # and 0 elsewhere.
    
    # Now, for the block-diagonal embedding:
    # 2x4x5 = (2x4x4) ⊕ (2x4x1)
    # Where 2x4x4 uses k in 0..3 and 2x4x1 uses k=4.
    
    # For 2x4x4: T[i*4+j, j*4+k, k*2+i] = 1 for i in 0..1, j in 0..3, k in 0..3
    # For 2x4x1: T[i*4+j, j*5+4, 4*2+i] = 1 for i in 0..1, j in 0..3
    
    # Note: in the 2x4x4 subproblem, the b-index is j*4+k (using p=4),
    # but in the full 2x4x5 problem, b = j*5+k. So the indices don't align directly.
    # The block-diagonal embedding needs careful index mapping.
    
    # Actually, the "direct sum" means we treat the k=0..3 part and k=4 part separately.
    # For k in 0..3: a = i*4+j, b = j*5+k, c = k*2+i
    # For k = 4: a = i*4+j, b = j*5+4, c = 4*2+i = 8+i
    
    # These are disjoint sets of (a,b,c) triples, so the tensor is indeed a direct sum.
    # We can decompose each part independently and concatenate the rank-1 terms.
    
    # Part 1 (k=0..3): 2x4x4-like but with b-index j*5+k instead of j*4+k
    # Part 2 (k=4): 2x4x1-like
    
    # For Part 2: T[i*4+j, j*5+4, 8+i] = 1 for i in 0..1, j in 0..3
    # This is 8 entries. Can we decompose with rank 8?
    # Each entry is independent, so schoolbook gives rank 8.
    # Can we do better? T[a,b,c] where a=i*4+j, b=j*5+4, c=8+i
    # For fixed j, b is fixed (b=j*5+4). For fixed i, c is fixed (c=8+i).
    # T[i*4+j, j*5+4, 8+i] = 1
    # This can be written as: sum_r U_r[i*4+j] * V_r[j*5+4] * W_r[8+i]
    # For each (i,j), we need the sum to be 1.
    # If we use U_r[i*4+j] = delta_{r, i*4+j}, V_r[j*5+4] = 1, W_r[8+i] = 1,
    # then for term r = i*4+j: U_r[i*4+j]=1, V_r[j*5+4]=1, W_r[8+i]=1.
    # But V_r[j'*5+4] for j' != j would also be 1, causing cross-terms.
    # So we need V_r[j*5+4] = 1 only for the correct j.
    # This means we need separate terms for each (i,j), giving rank 8.
    
    # So Part 2 has rank 8.
    
    # For Part 1 (k=0..3): 16 entries (2*4*4 = 32? No, 2*4*4=32 entries but many are 0)
    # Actually, for each (i,j,k) with i in 0..1, j in 0..3, k in 0..3,
    # we have T[i*4+j, j*5+k, k*2+i] = 1.
    # That's 2*4*4 = 32 entries.
    # Schoolbook gives rank 32 for this part.
    # We want rank 24 for this part.
    
    # Hmm, this is getting complicated. Let me just implement the known rank-36 solution
    # and try to improve it with the search, as the current program does.
    
    # Actually, let me try a different approach. Let me use the current rank-36 solution
    # but with a more aggressive search for rank reduction.
    
    # Let me restart with the original rank-36 construction and improve the search.
    
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    # Pre-compute target tensor for fast verification
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify_decomposition_fast(U, V, W):
        # Use einsum for fast tensor reconstruction and comparison
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Result: (nm, mp, pn)
        T = np.einsum('ir, jr, kr -> ijk', U, V, W)
        return np.allclose(T, target, atol=1e-9)

    # Verify rank-36 first
    if not verify_decomposition_fast(U, V, W):
        # Fallback to schoolbook if Strassen construction failed (should not happen)
        return best_result

    # Search for rank-35 with more alphas and random perturbations
    found_rank35 = False
    U35 = None
    V35 = None
    W35 = None
    
    alphas = [-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    
    # Try all pairs of columns with more alphas
    for i_col in range(total_rank):
        for j_col in range(i_col + 1, total_rank):
            if time.time() > deadline - 20:
                break
            for alpha in alphas:
                if time.time() > deadline - 10:
                    break
                # Create copies
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                
                # Merge column j_col into i_col: U_i += alpha * U_j
                U_test[:, i_col] += alpha * U_test[:, j_col]
                V_test[:, i_col] += alpha * V_test[:, j_col]
                W_test[:, i_col] += alpha * W_test[:, j_col]
                
                # Remove column j_col
                U_test = np.delete(U_test, j_col, axis=1)
                V_test = np.delete(V_test, j_col, axis=1)
                W_test = np.delete(W_test, j_col, axis=1)
                
                if verify_decomposition_fast(U_test, V_test, W_test):
                    found_rank35 = True
                    U35 = U_test
                    V35 = V_test
                    W35 = W_test
                    break
            if found_rank35 or time.time() > deadline - 10:
                break
        if found_rank35 or time.time() > deadline - 20:
            break
        
    if found_rank35:
        # Round to nearest half-integer
        U35_list = [[int(round(U35[i, r] * 2)) / 2 for r in range(35)] for i in range(nm)]
        V35_list = [[int(round(V35[i, r] * 2)) / 2 for r in range(35)] for i in range(mp)]
        W35_list = [[int(round(W35[i, r] * 2)) / 2 for r in range(35)] for i in range(pn)]
        return {"U": U35_list, "V": V35_list, "W": W35_list}

    # If rank 35 not found, try triple merges for rank 34
    found_rank34 = False
    U34 = None
    V34 = None
    W34 = None
    
    # Try merging 3 columns into 1
    # This is expensive, so limit the search
    if time.time() < deadline - 30:
        import itertools
        # Try all triples (i, j, k) with i < j < k
        # Merge j and k into i with coefficients alpha, beta
        for i_col in range(total_rank):
            if time.time() > deadline - 30:
                break
            for j_col in range(i_col + 1, total_rank):
                if time.time() > deadline - 30:
                    break
                for k_col in range(j_col + 1, total_rank):
                    if time.time() > deadline - 20:
                        break
                    # Try a few alpha, beta combinations
                    for alpha in [-1.0, -0.5, 0.5, 1.0]:
                        for beta in [-1.0, -0.5, 0.5, 1.0]:
                            if time.time() > deadline - 20:
                                break
                            U_test = U.copy()
                            V_test = V.copy()
                            W_test = W.copy()
                            
                            U_test[:, i_col] += alpha * U_test[:, j_col] + beta * U_test[:, k_col]
                            V_test[:, i_col] += alpha * V_test[:, j_col] + beta * V_test[:, k_col]
                            W_test[:, i_col] += alpha * W_test[:, j_col] + beta * W_test[:, k_col]
                            
                            # Remove j_col and k_col (remove higher index first)
                            if k_col > j_col:
                                U_test = np.delete(U_test, k_col, axis=1)
                                V_test = np.delete(V_test, k_col, axis=1)
                                W_test = np.delete(W_test, k_col, axis=1)
                                # j_col index unchanged since k_col > j_col
                            else:
                                U_test = np.delete(U_test, j_col, axis=1)
                                V_test = np.delete(V_test, j_col, axis=1)
                                W_test = np.delete(W_test, j_col, axis=1)
                                # k_col index shifts down by 1
                                k_col -= 1
                                U_test = np.delete(U_test, k_col, axis=1)
                                V_test = np.delete(V_test, k_col, axis=1)
                                W_test = np.delete(W_test, k_col, axis=1)
                            
                            if verify_decomposition_fast(U_test, V_test, W_test):
                                found_rank34 = True
                                U34 = U_test
                                V34 = V_test
                                W34 = W_test
                                break
                        if found_rank34:
                            break
                    if found_rank34:
                        break
                if found_rank34:
                    break
            if found_rank34:
                break
    
    if found_rank34:
        U34_list = [[int(round(U34[i, r] * 2)) / 2 for r in range(34)] for i in range(nm)]
        V34_list = [[int(round(V34[i, r] * 2)) / 2 for r in range(34)] for i in range(mp)]
        W34_list = [[int(round(W34[i, r] * 2)) / 2 for r in range(34)] for i in range(pn)]
        return {"U": U34_list, "V": V34_list, "W": W34_list}

    # If rank 34 not found, return rank 36
    U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
    V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
    W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
