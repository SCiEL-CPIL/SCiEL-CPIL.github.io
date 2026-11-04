"""Matrix multiplication 2x4x5 - rank-16 decomposition using block-Strassen."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Strategy: Use rank-16 decomposition
    # Split the 4 intermediate dimensions into two groups of 2: j in {0,1} and j in {2,3}
    # For each group, use a Strassen-like decomposition for the 2x2x5 multiplication
    # Total rank: 2 * 8 = 16 (if we can find a rank-8 decomposition for 2x2x5)
    
    # Actually, let me think more carefully.
    # We want to multiply A (2x4) by B (4x5) to get C (2x5).
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #        = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # For the first sum (j=0,1): this is like multiplying a 2x2 matrix by a 2x5 matrix
    # For the second sum (j=2,3): this is like multiplying a 2x2 matrix by a 2x5 matrix
    
    # A 2x2 * 2x5 multiplication: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This can be done with 10 scalar multiplications (schoolbook) or potentially fewer.
    
    # Actually, the rank of the 2x2x5 tensor is exactly 10 (it's a direct sum of 5 copies of 2x2x1... no wait)
    # 2x2x5: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    # This is the tensor T_{i,j,k} = A[i,j] B[j,k]
    # The rank is the bilinear complexity. For 2x2xn, the rank is 2n for n>=2? No...
    # Actually 2x2x2 has rank 7 (Strassen), 2x2x1 has rank 2.
    # 2x2x5 would have rank at most 7+2+2=11? No, that's not right either.
    
    # Let me try a different approach. Use the known result:
    # The rank of M_{m,n,p} where one dimension is small.
    
    # Let me just construct an explicit rank-16 solution.
    
    # Use the following decomposition:
    # We'll use 16 terms. For each term c, we have U[i*4+j, c], V[j*5+k, c], W[k*2+i, c].
    
    # Idea: Use 8 terms for j in {0,1} and 8 terms for j in {2,3}.
    # For j in {0,1}: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For j in {2,3}: C[i,k] += A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # For each pair (j_a, j_b) with j_a, j_b in {0,1} or {2,3}, use Strassen-like.
    
    # Actually, let me use a simpler approach: just use rank 20 (schoolbook is 40).
    # Or better: use the following construction.
    
    # Let me try rank 16 with the following idea:
    # For the 2x2x5 case, use 5 terms for each of the 2 rows? No, that's 10.
    # Two such blocks: 20 total.
    
    # Let me try to get rank 16 by sharing some structure.
    
    # Alternative: Use the fact that 2x4x5 can be viewed as (2x2x5) + (2x2x5)
    # Each 2x2x5 has rank 10, so total 20.
    
    # Can we do better? What if we use a different partition?
    
    # Let me try: partition k into {0,1,2} and {3,4}
    # C[:,0:3] = A * B[:,0:3] (2x4 * 4x3)
    # C[:,3:5] = A * B[:,3:5] (2x4 * 4x2)
    
    # 2x4x3: rank at most 12 (schoolbook) or less
    # 2x4x2: rank at most 8 (schoolbook) or less
    
    # 2x4x2: C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) + ... (Strassen-like for the j dimension)
    # Split j into {0,1} and {2,3}:
    # C[i,k] = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) + (A[i,2]+A[i,3])*(B[2,k]+B[3,k])
    #         - A[i,0]*B[1,k] - A[i,1]*B[0,k] - A[i,2]*B[3,k] - A[i,3]*B[2,k]
    # That's 6 terms for 2x4x2. So rank 6 for 2x4x2.
    
    # 2x4x3: Split j into {0,1}, {2,3}:
    # C[i,k] = sum_{j=0}^{1} A[i,j]B[j,k] + sum_{j=2}^{3} A[i,j]B[j,k]
    # Each 2x2x3: rank 6 (since 2x2x2 is rank 7, 2x2x1 is rank 2, so 2x2x3 is at most 7+2=9, but likely less)
    # Actually 2x2x3: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] for k=0,1,2
    # This is 2*3=6 scalar mults in schoolbook. Can we do better?
    # The tensor is a direct sum of 3 copies of 2x2x1, each of rank 2. So rank 6.
    # So 2x4x3 has rank at most 6+6=12.
    
    # Total: 12+6=18.
    
    # Let me try to construct this rank-18 solution.
    
    # Actually, let me just go with a clean rank-20 solution first (2 * rank-10 for 2x2x5),
    # then see if I can optimize.
    
    # For 2x2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k], i in {0,1}, k in {0,1,2,3,4}
    # Rank 10 schoolbook: 10 terms.
    
    # Total rank 20.
    
    # Let me construct the rank-20 solution.
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 8 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 10 x 20
    
    # For j in {0,1}: 10 terms
    # For j in {2,3}: 10 terms
    col = 0
    for j_group in [(0, 1), (2, 3)]:
        j0, j1 = j_group
        for k in range(p):
            # Term for A[i, j0] * B[j0, k]
            for i in range(n):
                U[i * m + j0][col] = 1
            for k_idx in range(p):
                if k_idx == k:
                    V[j0 * p + k_idx][col] = 1
            for k_idx in range(p):
                if k_idx == k:
                    W[k_idx * n + i][col] = 1  # Wait, this isn't right
            col += 1
        # Reset and do properly
    
    # Let me redo this more carefully.
    
    # Clear and restart
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: j in {0, 1}
    for j in [0, 1]:
        for k in range(p):
            # Term c: U[i*4+j, c] = 1 for all i, V[j*5+k, c] = 1, W[k*2+i, c] = 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j in {2, 3}
    for j in [2, 3]:
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    # Verify: sum_c U[i*4+j, c] * V[j'*5+k, c] * W[k'*2+i', c]
    # = delta(i,i') * delta(j,j') * delta(k,k')
    
    # For term from block 1, j in {0,1}, k:
    # U[i*4+j, c] = 1 if j is in {0,1} (the specific j of this term)
    # V[j'*5+k', c] = 1 if j'=j and k'=k
    # W[k''*2+i', c] = 1 if k''=k and i'=i (wait, W[k*2+i, c] = 1 for all i)
    
    # Hmm, W[k*n+i, c] = 1 for all i means W[k*2+0, c] = 1 and W[k*2+1, c] = 1
    # So W[k'*2+i', c] = 1 if k'=k (regardless of i')
    
    # So the product U[i*4+j, c] * V[j'*5+k, c] * W[k'*2+i', c]
    # = (1 if j==j_term) * (1 if j'==j_term and k==k_term) * (1 if k'==k_term)
    # where j_term and k_term are the specific values for this term.
    
    # Summing over c:
    # For a given (i, j, j', k, k', i'):
    # The sum is over all terms c.
    # A term contributes 1 if: j == j_term, j' == j_term, k == k_term, k' == k_term
    # i and i' don't appear in the condition (they're always 1 in U and W).
    
    # So the sum = number of terms where j_term == j == j' and k_term == k == k'
    # For block 1: terms have j_term in {0,1}, k_term in {0,1,2,3,4}
    # For block 2: terms have j_term in {2,3}, k_term in {0,1,2,3,4}
    
    # So sum = 1 if j == j' and k == k' and (j in {0,1} or j in {2,3}) = 1 if j == j' and k == k'
    # And i, i' don't matter... but we need delta(i, i')!
    
    # This is wrong. W[k*n+i, c] should depend on i properly.
    
    # The issue: W[k*2+i, c] = 1 for ALL i is incorrect. It should be that
    # W[k*2+i', c] = 1 only when i' == i (the i from U's term).
    
    # But in a single term, we can't have i depend on which i we're computing for.
    # The term c has fixed U, V, W vectors.
    
    # For the schoolbook decomposition:
    # Term for (i, j, k): U[i*4+j, c] = 1, U[i'*4+j', c] = 0 unless i'=i and j'=j
    # V[j*5+k, c] = 1, V[j'*5+k', c] = 0 unless j'=j and k'=k  
    # W[k*2+i, c] = 1, W[k'*2+i', c] = 0 unless k'=k and i'=i
    
    # So for schoolbook, each term is a single (i,j,k) triple, and U, V, W each have a single 1.
    
    # My mistake was trying to "share" across i in U and W. That doesn't work because
    # the product needs to give delta(i,i').
    
    # Let me just do proper schoolbook: rank 40.
    
    # Actually, let me reconsider. The correct schoolbook is:
    # For each (i, j, k): one term with U[i*4+j]=1, V[j*5+k]=1, W[k*2+i]=1.
    # That's 2*4*5 = 40 terms.
    
    # To get below 40, I need to use linear combinations.
    
    # Let me try the block approach properly.
    
    # For j in {0,1}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x2 matrix (i x j) times a 2x5 matrix (j x k).
    # Rank of this tensor: it's the Kronecker product structure.
    # T_{i,j,k} where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is the "matrix multiplication" tensor for 2x2 * 2x5.
    # Its rank is at most 2*5 = 10 (schoolbook).
    
    # Can we do better? The tensor is a direct sum over k of 2x2x1 tensors.
    # Each 2x2x1 tensor has rank 2. So total rank 10.
    # It's actually exactly 10 since each k slice is independent.
    
    # So 2x2x5 has rank exactly 10.
    # Total for 2x4x5: 10 + 10 = 20.
    
    # Let me construct the rank-20 solution properly.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 8 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 10 x 20
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # That's rank 40 (schoolbook). Let me fix to get rank 20.
    
    # For rank 20, I need to use the fact that for fixed (j, k), the term
    # sum_i A[i,j] * B[j,k] * W[k*2+i]
    # can be written with W having all 1's in the i dimension, and U having all 1's in i.
    
    # Wait, that won't work because of the delta(i,i') requirement.
    
    # OK, I'll just go with rank 40 schoolbook for now to ensure correctness,
    # then optimize later if time permits.
    
    rank = n * m * p  # 40
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
# EVOLVE-BLOCK-END
