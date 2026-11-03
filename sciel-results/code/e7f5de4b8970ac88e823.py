"""Tensor decomposition for 2x3 matrix times 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x6 matrix.
    
    The multiplication tensor T has entries T[i, j, k] where:
      i in {0,1} (row of A, row of C)
      j in {0,1,2} (column of A / row of B)
      k in {0,1,2,3,4,5} (column of B, column of C)
    
    The factorization is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where m=3, n=2, p=6.
    
    We use a structured approach: for each k (6 values), we decompose the
    2x3 "slice" using a rank-3 construction (one per j), but we exploit
    linearity across k to share terms.
    
    Actually, the key insight: for fixed k, we need to compute C[:,k] = A @ B[:,k].
    This is a 2x3 matrix times a 3-vector. The tensor for this is rank 3 (one per j).
    But across all 6 k values, naively that's 18 terms for the "inner products".
    However, we also need to separate the two rows of C.
    
    Let me think more carefully. The full tensor T[i,j,k] = delta (it's the identity
    multiplication tensor). We want:
      sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*2+i, c] = 1 for all valid (i,j,k)
    
    A standard approach: use 2*3*6 = 36 terms (schoolbook).
    
    To reduce rank, we can use the fact that for a 2x3 times 3x6 multiplication,
    we can group by k. For each k, the contribution to C[0,k] is sum_j A[0,j]*B[j,k]
    and C[1,k] is sum_j A[1,j]*B[j,k].
    
    Key optimization: For each k, instead of 3 separate terms for each row of C,
    we can use 3 terms that contribute to BOTH rows simultaneously if we set
    U[i*3+j, c] appropriately. But we need U[0*3+j, c] = 1 and U[1*3+j, c] = 1
    for the same term, which would give us A[0,j]*B[j,k] added to C[1,k] as well.
    
    So we need to use linear combinations. Let's use the following strategy:
    
    For each k in 0..5:
      For each j in 0..2:
        We need a term that gives A[0,j]*B[j,k] to C[0,k] and A[1,j]*B[j,k] to C[1,k].
        
    The minimal approach for 2x3x1 (2x3 times 3x1) has rank 3 (one per j).
    For 2x3x6, if we could share, we'd get 3 terms total, but that's not possible
    because the B values differ per k.
    
    However, we can use a Strassen-like trick for the 2x3 part. The rank of the
    2x3x3 matrix multiplication tensor is 7 (known result). For 2x3x6, we can
    potentially use 7 terms per "group" of 3 k-values, giving 14 terms for 6 k-values?
    No, that doesn't quite work because the k dimension is independent.
    
    Let me try a direct construction with rank 20:
    
    Idea: Use 20 terms. For each of the 6 k-values, allocate about 3.33 terms.
    We can use 4 terms for some k-values and 3 for others, or use a more clever
    sharing.
    
    Actually, let me try a different approach. The tensor for 2x3x6 multiplication
    can be viewed as 6 copies of the 2x3x1 tensor (one per k), but the B factors
    are different. 
    
    Let me try rank = 20 with the following construction:
    
    For k in 0..5, use 3 terms each = 18 terms, plus 2 extra terms for cross-coupling.
    
    Actually, let me just construct a valid decomposition. The simplest correct
    approach is schoolbook rank 36, but let me try to do better.
    
    Here's a rank-20 construction:
    
    We'll use the following insight: for the 2x3 part, we can use the fact that
    2*3 = 6, and 6 = 3+3. We can split the j dimension into two groups.
    
    Let me try a concrete construction:
    
    Terms 0-5: For j=0, k=0..5, contribute to both rows
      U[0*3+0, c] = 1, U[1*3+0, c] = 1
      V[0*6+k, c] = 1
      W[k*2+0, c] = 1, W[k*2+1, c] = 1
    This gives A[0,0]*B[0,k] to C[0,k] AND A[1,0]*B[0,k] to C[1,k], but also
    A[0,0]*B[0,k] to C[1,k] and A[1,0]*B[0,k] to C[0,k] which is WRONG.
    
    So that doesn't work. We need to separate the rows.
    
    Correct approach: For each (i, j, k) we need exactly one term, OR we use
    linear combinations that cancel out the unwanted contributions.
    
    Let me use the standard rank-36 schoolbook for correctness, but I'll try
    to find a lower rank through optimization.
    
    Actually, for the purpose of this task, let me construct a valid rank-20
    decomposition using the following method:
    
    We observe that C[i,k] = sum_j A[i,j] * B[j,k].
    
    For each k, define vectors a0_k = (A[0,0], A[0,1], A[0,2]) and 
    a1_k = (A[1,0], A[1,1], A[1,2]), and b_k = (B[0,k], B[1,k], B[2,k]).
    
    C[0,k] = a0 . b_k, C[1,k] = a1 . b_k.
    
    We can write this as:
    C[0,k] = (a0 + a1) . b_k - a1 . b_k
    C[1,k] = (a0 + a1) . b_k - a0 . b_k
    
    So for each k, we need 3 terms: one for (a0+a1).b_k, one for a0.b_k, one for a1.b_k.
    That's 3*6 = 18 terms. But we can share the (a0+a1).b_k term? No, because
    b_k is different for each k.
    
    So 18 terms minimum with this approach. Let me see if we can do 18.
    
    For each k in 0..5:
      Term 3k:   U[0*3+j] = 1 for all j, U[1*3+j] = 1 for all j  -> (a0+a1) part
                 V[j*6+k] = 1 for all j
                 W[k*2+0] = 1, W[k*2+1] = 1
                 This contributes (a0+a1).b_k to both C[0,k] and C[1,k]
                 
      Term 3k+1: U[0*3+j] = 1 for all j, U[1*3+j] = 0 for all j  -> a0 part
                 V[j*6+k] = 1 for all j
                 W[k*2+0] = -1, W[k*2+1] = 0
                 This contributes -a0.b_k to C[0,k]
                 
      Term 3k+2: U[0*3+j] = 0 for all j, U[1*3+j] = 1 for all j  -> a1 part
                 V[j*6+k] = 1 for all j
                 W[k*2+0] = 0, W[k*2+1] = -1
                 This contributes -a1.b_k to C[1,k]
    
    Let's verify:
    C[0,k] = (a0+a1).b_k - a0.b_k = a1.b_k. WRONG! Should be a0.b_k.
    
    Let me fix:
    C[0,k] = (a0+a1).b_k - a1.b_k = a0.b_k  ✓
    C[1,k] = (a0+a1).b_k - a0.b_k = a1.b_k  ✓
    
    So:
      Term 3k:   U[i*3+j] = 1 for all i,j; V[j*6+k] = 1; W[k*2+0] = 1, W[k*2+1] = 1
      Term 3k+1: U[0*3+j] = 0, U[1*3+j] = 1; V[j*6+k] = 1; W[k*2+0] = -1, W[k*2+1] = 0
      Term 3k+2: U[0*3+j] = 1, U[1*3+j] = 0; V[j*6+k] = 1; W[k*2+0] = 0, W[k*2+1] = -1
    
    Wait, let me re-derive:
    We want C[0,k] = a0.b_k and C[1,k] = a1.b_k.
    
    Let s_k = (a0+a1).b_k.
    C[0,k] = s_k - a1.b_k
    C[1,k] = s_k - a0.b_k
    
    So:
      Term for s_k: contributes +1 to both C[0,k] and C[1,k]
        U[i*3+j] = 1 for all i in {0,1}, j in {0,1,2}
        V[j*6+k] = 1 for all j
        W[k*2+0] = 1, W[k*2+1] = 1
      
      Term for -a1.b_k: contributes -1 to C[0,k], 0 to C[1,k]
        U[0*3+j] = 1, U[1*3+j] = 0 for all j  (this is a0, not a1!)
        
    I'm confusing myself. Let me be very explicit.
    
    a0 = (A[0,0], A[0,1], A[0,2]), a1 = (A[1,0], A[1,1], A[1,2]).
    b_k = (B[0,k], B[1,k], B[2,k]).
    
    s_k = (a0 + a1) . b_k = sum_j (A[0,j] + A[1,j]) * B[j,k]
    
    C[0,k] = a0 . b_k = s_k - a1 . b_k
    C[1,k] = a1 . b_k = s_k - a0 . b_k
    
    Term 3k (for s_k):
      U[i*3+j, 3k] = 1 for all i in {0,1}, j in {0,1,2}
      V[j*6+k, 3k] = 1 for all j in {0,1,2}
      W[k*2+0, 3k] = 1
      W[k*2+1, 3k] = 1
    
    Term 3k+1 (for -a1.b_k, to subtract from C[0,k]):
      U[0*3+j, 3k+1] = 0 for all j  (no contribution from row 0 of A)
      U[1*3+j, 3k+1] = 1 for all j  (contribution from row 1 of A)
      V[j*6+k, 3k+1] = 1 for all j
      W[k*2+0, 3k+1] = -1  (subtract from C[0,k])
      W[k*2+1, 3k+1] = 0
    
    Term 3k+2 (for -a0.b_k, to subtract from C[1,k]):
      U[0*3+j, 3k+2] = 1 for all j  (contribution from row 0 of A)
      U[1*3+j, 3k+2] = 0 for all j
      V[j*6+k, 3k+2] = 1 for all j
      W[k*2+0, 3k+2] = 0
      W[k*2+1, 3k+2] = -1  (subtract from C[1,k])
    
    Let's verify for C[0,k]:
    sum_c U[0*3+j, c] * V[j*6+k, c] * W[k*2+0, c]
    = sum_j [U[0*3+j, 3k]*V[j*6+k, 3k]*W[k*2+0, 3k] 
           + U[0*3+j, 3k+1]*V[j*6+k, 3k+1]*W[k*2+0, 3k+1]
           + U[0*3+j, 3k+2]*V[j*6+k, 3k+2]*W[k*2+0, 3k+2]]
    = sum_j [1*1*1 + 0*1*(-1) + 1*1*0]
    = sum_j [1 + 0 + 0]
    = 3
    
    That's not right. We need it to equal sum_j A[0,j]*B[j,k], not 3.
    
    The issue is that U and V should encode the A and B values, not be fixed to 1.
    But in the tensor decomposition, U, V, W are fixed matrices that work for ALL
    input matrices A and B. So U[i*3+j, c] is a constant, not depending on A.
    
    The correct interpretation: the tensor T[i,j,k] = 1 (it's the identity tensor for
    multiplication). The factorization is:
    T[i,j,k] = sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*2+i, c]
    
    And T[i,j,k] = 1 for all valid (i,j,k) because the multiplication tensor has
    all entries equal to 1 in the "diagonal" sense... wait, no.
    
    Actually, the matrix multiplication tensor for (n,m,p) has entries:
    T[i,j,k] where i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    and T[i,j,k] = 1 if the indices are "compatible" for the multiplication.
    
    For standard matrix multiplication C = A*B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has T[i,j,k] = 1 for all i, j, k (it's a 3-way tensor of all ones
    in the relevant indices). No wait, that's not right either.
    
    The multiplication tensor is defined as:
    T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * (1)
    
    where the 1 indicates that A[i,j] * B[j,k] contributes to C[i,k].
    
    So T[i,j,k] = 1 for all valid i, j, k. The factorization is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    
    So we need: for every (i, j, k) triple, the sum equals 1.
    
    Let me redo the construction with this understanding.
    
    For the term 3k:
    Contribution to T[i,j,k] = U[i*3+j, 3k] * V[j*6+k, 3k] * W[k*2+i, 3k]
    
    For term 3k to contribute to T[i,j,k], we need:
    U[i*3+j, 3k] * V[j*6+k, 3k] * W[k*2+i, 3k]
    
    If we set U[i*3+j, 3k] = 1 for all i, j, V[j*6+k, 3k] = 1 for all j,
    W[k*2+i, 3k] = 1 for all i, then term 3k contributes 1 to T[i,j,k] for all i, j.
    
    But we only want it to contribute to specific (i,j) pairs.
    
    I think I was overcomplicating this. Let me just use the schoolbook decomposition
    with rank 36, which is guaranteed correct.
    
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook decomposition: rank = n*m*p = 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 18 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
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
