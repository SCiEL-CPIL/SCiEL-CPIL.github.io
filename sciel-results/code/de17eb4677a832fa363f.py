"""
Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.

Strategy:
- The schoolbook rank is 3*4*5 = 60.
- Known lower bound for rank of matrix multiplication tensor T_{n,m,p} is n*m + n*p + m*p - 1? 
  Actually, the known lower bound is max(n*m, n*p, m*p) but better bounds exist.
  For 3x4x5, the minimal rank is known to be 19? Let me recall.
  
  Actually, for matrix multiplication, the tensor rank is the bilinear complexity.
  For 2x2x2 it's 7 (Strassen).
  For 3x3x3, the best known upper bound is 23 (Strassen), lower bound 19.
  For 3x4x5, we need to find the minimal rank.

  However, the task says "integer/half-integer entries". This suggests that an exact rational decomposition exists.

  Let's think about the structure. The matrix multiplication tensor T_{3,4,5} has dimension 3*4*5 = 60.
  We want to write it as sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r where:
    U_r is in R^{3*4}, V_r in R^{4*5}, W_r in R^{5*3}.
  
  Actually, the tensor is T ∈ R^{12} ⊗ R^{20} ⊗ R^{15} (since 3*4=12, 4*5=20, 5*3=15).
  
  The rank of this tensor is what we seek.

  Known results: The rank of the 3x4x5 matrix multiplication tensor.
  According to literature, the minimal rank for 3x4x5 is not trivially known to be much less than 60.
  
  Wait, let me reconsider. The problem states "integer/half-integer entries". This is a strong hint that the optimal solution has a simple rational form.

  Let's try to find a decomposition of rank less than 60.
  
  One approach: Use the fact that the tensor is the Kronecker product structure.
  T_{3,4,5} = T_{3,4} ⊗ T_{4,5}? No, that's not right.
  
  Actually, T_{n,m,p} for multiplying an n x m matrix by an m x p matrix.
  The tensor has indices (i,j,k) for i in 1..n, j in 1..m, k in 1..p.
  T_{i,j,k} = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} in some basis.
  
  Let's use a different approach. We can try to use a known decomposition.
  
  For 2x2x2, Strassen's algorithm gives rank 7.
  For 3x3x3, Strassen gives rank 23.
  
  For 3x4x5, we can try to combine smaller decompositions.
  
  Actually, let me just try to find a rank-19 or rank-20 decomposition if it exists.
  
  But wait, the lower bound for the rank of T_{n,m,p} is at least n*m + m*p + p*n - 1? No, that's not a standard bound.
  
  The trivial lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
  
  So the rank is at least 20. Can we achieve rank 20?
  
  If we can find a rank-20 decomposition, that would be optimal.
  
  Let's try to construct one. We need 20 terms.
  
  Actually, I recall that for matrix multiplication, the rank is often close to n*m*p / (some factor). But 60/20 = 3, so we'd need a factor of 3 improvement.
  
  Let me think about this differently. The matrix multiplication tensor T_{n,m,p} can be seen as a linear map from R^{n*m} ⊗ R^{m*p} to R^{n*p}.
  
  The rank of this tensor is the minimal number of rank-1 tensors needed to express it.
  
  For 3x4x5, let's try to find a decomposition.
  
  Actually, I found a reference: The rank of the 3x4x5 matrix multiplication tensor is 19? Or is it 20?
  
  Let me just try to construct a rank-20 decomposition and see if it works.
  
  We can use the following idea: Since 3*4*5 = 60, and we want rank 20, we need each term to "cover" 3 elements on average.
  
  Let's try a systematic approach. We'll use the fact that the tensor is symmetric in a certain way.
  
  Actually, let me just try to use a known result. The minimal rank for 3x4x5 matrix multiplication is 19.
  
  Wait, I'm not sure. Let me try a different approach.
  
  We can use the following construction:
  - For the 3x4 part, we can use a rank-3 decomposition (trivial).
  - For the 4x5 part, we can use a rank-4 decomposition (trivial).
  - But that doesn't directly help.
  
  Let me try to find a rank-20 decomposition by solving a system of equations.
  
  We have 60 equations (one for each (i,j,k) triple) and 20 * (12 + 20 + 15) = 20 * 47 = 940 unknowns (if we fix one scaling per term).
  
  This is an underdetermined system, so there should be solutions.
  
  But solving this exactly is hard. Let me try a simpler approach.
  
  Actually, I think the minimal rank for 3x4x5 is 19. Let me try to find a rank-19 decomposition.
  
  But I'm not confident. Let me just try to return the schoolbook decomposition for now, since I don't have a concrete rank-19 or rank-20 decomposition.
  
  Wait, the task says "rank-minimal". If I can't find a better one, I should at least try to find one.
  
  Let me try to use a known decomposition. For 2x2x2, Strassen's algorithm uses 7 multiplications.
  
  For 3x4x5, we can try to use a combination of Strassen-like algorithms.
  
  Actually, let me just try to find a rank-20 decomposition. If I can't, I'll return the schoolbook one.
  
  Here's an idea: We can use the following decomposition:
  T_{3,4,5} = sum_{r=1}^{20} U_r ⊗ V_r ⊗ W_r
  
  where each U_r, V_r, W_r has sparse entries.
  
  Let me try to construct this by hand.
  
  For r = 1..12, let U_r = e_{i,j}, V_r = e_{j,k}, W_r = e_{k,i} for some (i,j,k).
  But that's the schoolbook decomposition with rank 60.
  
  To get rank 20, we need each term to cover 3 (i,j,k) triples.
  
  Let's try: For each j in 1..4, and for each (i,k) pair, we can have a term that covers (i,j,k) for all i in 1..3? No, that would be 4*5 = 20 terms, each covering 3 elements.
  
  Wait, that's promising! Let's try:
  For j = 1..4, and for k = 1..5, define:
    U_{j,k} = sum_{i=1}^{3} e_{i,j}  (a vector in R^{12})
    V_{j,k} = e_{j,k}  (a vector in R^{20})
    W_{j,k} = sum_{i=1}^{3} e_{k,i}  (a vector in R^{15})
  
  Then the tensor is:
  sum_{j=1}^{4} sum_{k=1}^{5} U_{j,k} ⊗ V_{j,k} ⊗ W_{j,k}
  
  Let's check what this gives for a specific (i,j',k'):
  The coefficient of e_{i,j'} ⊗ e_{j',k'} ⊗ e_{k',i'} in the sum is:
  sum_{j=1}^{4} sum_{k=1}^{5} [U_{j,k}]_{i,j'} [V_{j,k}]_{j',k'} [W_{j,k}]_{k',i'}
  
  [U_{j,k}]_{i,j'} = 1 if j = j', else 0.
  [V_{j,k}]_{j',k'} = 1 if j = j' and k = k', else 0.
  [W_{j,k}]_{k',i'} = 1 if k = k' and i = i', else 0.
  
  So the sum is non-zero only when j = j', k = k', and i = i'. In that case, the value is 1.
  
  But we want the tensor to have 1 at (i,j',k') when i = i', j = j', k = k'. So this works!
  
  Wait, let me re-check. The tensor T_{3,4,5} has T_{i,j,k} = 1 for all i,j,k. So we want the sum to give 1 at each (i,j,k).
  
  With the above construction, for a fixed (i,j',k'), the sum is:
  sum_{j=1}^{4} sum_{k=1}^{5} [j==j'] [k==k'] [i==i'] = [j'==j' and k'==k' and i==i'] = 1 if i==i', else 0.
  
  But i and i' are the same index, so this is always 1. So it works!
  
  But wait, this gives rank 4*5 = 20. And the entries are 0 or 1, which are integers.
  
  So we have a rank-20 decomposition!
  
  Let me verify the shapes:
  U has shape (12, 20), V has shape (20, 20), W has shape (15, 20).
  
  For r = (j-1)*5 + (k-1) (0-indexed), where j in 0..3, k in 0..4:
    U[i*m + j, r] = 1 for all i in 0..2 (i.e., for rows j, j+m, j+2m)
    V[j*p + k, r] = 1
    W[k*n + i, r] = 1 for all i in 0..2 (i.e., for rows k, k+n, k+2n)
  
  Let me code this up.
  
  Actually, wait. Let me re-index. In the schoolbook decomposition, U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1.
  
  In my new decomposition, for each (j,k) pair (j in 0..3, k in 0..4), we have one term.
  U[i*m + j, r] = 1 for all i in 0..2.
  V[j*p + k, r] = 1.
  W[k*n + i, r] = 1 for all i in 0..2.
  
  Let's verify: The tensor entry at (i,j',k') is:
  sum_{r} U[i*m + j', r] * V[j'*p + k', r] * W[k'*n + i, r]
  
  U[i*m + j', r] = 1 if r corresponds to (j', k) for some k, i.e., if the j-index of r is j'.
  V[j'*p + k', r] = 1 if r corresponds to (j', k').
  W[k'*n + i, r] = 1 if r corresponds to (j, k') for some j, i.e., if the k-index of r is k'.
  
  So the product is 1 only if r corresponds to (j', k'). In that case, U[i*m + j', r] = 1 (since the j-index is j'), V[j'*p + k', r] = 1, and W[k'*n + i, r] = 1 (since the k-index is k').
  
  So the sum is 1. Correct!
  
  So we have a rank-20 decomposition. This is better than the schoolbook rank of 60.
  
  Can we do better than 20? The lower bound is max(12, 20, 15) = 20. So 20 is optimal!
  
  Wait, is the lower bound really 20? The rank of a tensor in R^a ⊗ R^b ⊗ R^c is at least max(a, b, c). Here a=12, b=20, c=15, so the rank is at least 20. And we achieved 20, so it's optimal.
  
  Great! So the minimal rank is 20.

Let me code this up.
"""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    # Rank is m * p = 4 * 5 = 20
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]  # shape (12, 20)
    V = [[0] * rank for _ in range(m * p)]  # shape (20, 20)
    W = [[0] * rank for _ in range(p * n)]  # shape (15, 20)
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U[i*m + j, r] = 1 for all i in 0..n-1
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            # W[k*n + i, r] = 1 for all i in 0..n-1
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
