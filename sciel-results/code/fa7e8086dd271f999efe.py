"""
Rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.

Strategy:
- The schoolbook rank is n*m*p = 48.
- We use a block-structured decomposition that exploits the small n=2.
- For each pair of rows i1, i2 in {0,1} and each column k in {0..5},
  we can potentially share computations across the m=4 intermediate dimension.
  
  Actually, let's think more carefully. The tensor is T[i, j, k] where
  i in [0,2), j in [0,4), k in [0,6).
  
  A standard lower bound: rank >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
  
  We can achieve rank 24 by using the fact that for each fixed j, the
  sub-tensor over (i, k) has rank at most min(n, p) = 2... no that's not right.
  
  Let me think differently. The bilinear map is:
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  For n=2, p=6, m=4:
  We have 2*6 = 12 output elements, each is a sum of 4 products.
  
  One approach: for each output element C[i,k], we compute sum_j A[i,j]*B[j,k].
  That's 12 * 4 = 48 multiplications (schoolbook).
  
  Can we do better? Yes, if we can share multiplications.
  
  Key insight: A[i,j] is shared across all k (6 values of k), and B[j,k] is shared
  across all i (2 values of i).
  
  For a fixed j, we have A[0,j], A[1,j] (2 values) and B[j,0..5] (6 values).
  The products are A[0,j]*B[j,k] and A[1,j]*B[j,k] for k=0..5. That's 12 products
  for each j, but we can write them as:
    A[0,j]*B[j,k] for k=0..5
    A[1,j]*B[j,k] for k=0..5
  
  If we precompute nothing, that's 12 multiplications per j, 4 j's = 48.
  
  But wait - can we use Strassen-like ideas? For 2x2 matrices, Strassen uses 7
  multiplications instead of 8. Here we have a 2x4 times 4x6.
  
  Let's think of it as: for each j, we have a "vector" a_j = (A[0,j], A[1,j]) in F^2
  and b_j = (B[j,0], ..., B[j,5]) in F^6. We need to compute:
    C[i,k] = sum_j a_j[i] * b_j[k]
  
  This is like computing a 2x6 matrix C = sum_j a_j * b_j^T where a_j is 2x1 and b_j is 1x6.
  
  The rank of this sum... each term a_j * b_j^T is a rank-1 2x6 matrix. We're summing 4 such terms.
  
  The bilinear complexity is the minimum number of scalar multiplications needed.
  
  For this specific structure, we can try to find a decomposition. Let's use the following approach:
  
  We want to find rank r and matrices U (8 x r), V (24 x r), W (12 x r) such that:
  T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
  
  Wait, the convention is W[k*n+i, r]. Let me re-read.
  
  "output factor follows the official trace convention W[k*n+i,r]"
  
  So W has indices (k*n + i) where k in [0,p), i in [0,n). That's p*n = 12 rows.
  U has indices (i*m + j) where i in [0,n), j in [0,m). That's n*m = 8 rows.
  V has indices (j*p + k) where j in [0,m), k in [0,p). That's m*p = 24 rows.
  
  The tensor entry T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
  
  Let's try to find a rank-24 decomposition. The lower bound is max(8,24,12)=24.
  
  Construction for rank 24:
  For each j in [0,4) and k in [0,6), define a "column" c = j*6 + k (0 to 23).
  Set:
    U[i*m + j, c] = 1 if... hmm, but U depends on i too.
    
  Actually, let's try:
    U[i*m + j, j*6+k] = 1 for all i? No, that would make U have the same value for different i.
    
  Let me try a different approach. We want:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is the "identity" tensor entry.
    
  The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k] and the tensor encodes the bilinear map.
  
  Actually, the tensor for matrix multiplication is: T[i,j,k] = 1 for all i,j,k in their ranges. Because the (i,k)-th output is sum_j A[i,j] B[j,k], so the tensor has 1s everywhere.
  
  So we need: for all i in [0,2), j in [0,4), k in [0,6):
    sum_c U[i*4+j, c] * V[j*6+k, c] * W[k*2+i, c] = 1
  
  And we want to minimize r = number of columns.
  
  Lower bound: r >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
  
  Can we achieve rank 24? Let's try:
  
  Set r = 24. For c = j*6 + k (j in 0..3, k in 0..5):
    U[i*4 + j, c] = 1 for all i? But then U[i*4+j, c] doesn't depend on i, which is fine.
    V[j*6 + k, c] = 1
    W[k*2 + i, c] = 1 for all i? 
    
    Then sum_c U[i*4+j,c] * V[j*6+k,c] * W[k*2+i,c] = sum_{j',k'} U[i*4+j, j'*6+k'] * V[j*6+k, j'*6+k'] * W[k*2+i, j'*6+k']
    
    U[i*4+j, j'*6+k'] = 1 if j' = j (since c = j'*6+k', and we set U[i*4+j, c]=1 only when the j-index matches). Wait, no. I said U[i*4+j, c] = 1 for all i, but c = j'*6+k'. So U[i*4+j, j'*6+k'] should be 1 if... 
    
    Let me be more careful. Let's define:
    For c = j'*6 + k' (j' in 0..3, k' in 0..5):
      U[i*4 + j, c] = 1 if j = j', else 0. (Doesn't depend on i or k')
      V[j*6 + k, c] = 1 if j = j' AND k = k', else 0.
      W[k*2 + i, c] = 1 if k = k', else 0. (Doesn't depend on i)
    
    Then the sum is:
    sum_{j',k'} [j=j'] * [j=j' and k=k'] * [k=k'] = sum_{j',k'} [j=j'] * [k=k'] = 1 (when j'=j, k'=k)
    
    So this works! And r = 24.
    
    But wait, this is just the schoolbook decomposition rearranged. The schoolbook has r = n*m*p = 48 with one column per (i,j,k). Here I've merged the i-dependence, getting r = m*p = 24.
    
    Is 24 achievable? The lower bound is 24, and I've constructed a rank-24 decomposition. So the answer is 24.
    
    Let me double-check the construction:
    r = 24
    For c in 0..23:
      j' = c // 6
      k' = c % 6
      For i in 0..1:
        U[i*4 + j', c] = 1
      V[j'*6 + k', c] = 1
      For i in 0..1:
        W[k'*2 + i, c] = 1
    
    Check: for fixed (i,j,k):
    sum_c U[i*4+j, c] * V[j*6+k, c] * W[k*2+i, c]
    = sum_{j',k'} U[i*4+j, j'*6+k'] * V[j*6+k, j'*6+k'] * W[k*2+i, j'*6+k']
    U[i*4+j, j'*6+k'] = 1 if j=j' else 0
    V[j*6+k, j'*6+k'] = 1 if j=j' and k=k' else 0
    W[k*2+i, j'*6+k'] = 1 if k=k' else 0
    
    So the product is 1 only when j=j' and k=k', giving sum = 1. Correct!
    
    So rank 24 is achievable. This is the minimum possible.
"""

import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = m * p = 24 (lower bound is max(n*m, m*p, p*n) = 24)
    rank = m * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    for c in range(rank):
        jp = c // p  # j' index
        kp = c % p   # k' index
        
        # U[i*m + j, c] = 1 if j == jp
        for i in range(n):
            U[i * m + jp][c] = 1
        
        # V[j*p + k, c] = 1 if j == jp and k == kp
        V[jp * p + kp][c] = 1
        
        # W[k*n + i, c] = 1 if k == kp
        for i in range(n):
            W[kp * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
