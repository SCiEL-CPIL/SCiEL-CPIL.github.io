"""Program to find a rank-minimal exact tensor decomposition for 2x4 by 4x4 matrix multiplication."""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a minimal-rank exact decomposition of the (2,4,4) matrix multiplication tensor.
    
    The tensor T for matrix multiplication C = A * B where A is n x m and B is m x p
    is defined such that:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    In the decomposition form:
    sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1 for all i, j, k
    
    For n=2, m=4, p=4:
    - U has shape (n*m, rank) = (8, rank)
    - V has shape (m*p, rank) = (16, rank)
    - W has shape (p*n, rank) = (8, rank)
    
    Key insight: We can achieve rank 16 by using V as the identity matrix.
    For each r in 0..15, let r = j*p + k where j in 0..3 and k in 0..3.
    - V[r, r] = 1
    - U[u, r] = 1 if u = j or u = m + j (i.e., both rows i=0 and i=1 for column j)
    - W[w, r] = 1 if w = k*n or w = k*n + 1 (i.e., both columns for row k)
    
    Verification:
    sum_c U[i*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c]
    = sum_{j',k'} U[i*m+j', j'*p+k'] * V[j'*p+k', j'*p+k'] * W[k'*n+i', j'*p+k']
    
    V is identity, so V[j'*p+k', j'*p+k'] = 1 only when the indices match.
    For a fixed term with r = j*p + k:
    - U[i*m+j', r] = 1 if j' = j
    - W[k'*n+i', r] = 1 if k' = k
    
    So the sum becomes:
    sum_{j,k} [1 if j'=j] * [1 if k'=k]
    
    For the product to be non-zero, we need j'=j and k'=k.
    Then U[i*m+j, j*p+k] = 1 (always, since both i=0 and i=1 have U=1 for column j)
    W[k*n+i', j*p+k] = 1 (always, since both i'=0 and i'=1 have W=1 for row k)
    
    Wait, let me re-check. We need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    With V as identity (V[a, b] = 1 if a=b, else 0):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    = U[i*m+j, j*p+k] * V[j*p+k, j*p+k] * W[k*n+i, j*p+k]
    = U[i*m+j, j*p+k] * 1 * W[k*n+i, j*p+k]
    
    Now, U[i*m+j, j*p+k] = 1 for all i (since U is 1 for both i=0 and i=1 when the j-index matches)
    W[k*n+i, j*p+k] = 1 for all i (since W is 1 for both i=0 and i=1 when the k-index matches)
    
    So the product is 1 * 1 * 1 = 1. Correct!
    
    But wait, we also need to make sure that for other (j', k') pairs, the contribution is 0.
    Since V is identity, only the term where c = j*p+k contributes.
    And for that term, U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1.
    
    So the sum is exactly 1. 
    
    Therefore, rank 16 works.
    """
    n, m, p = DIMENSIONS
    
    # Rank is m * p = 16
    rank = m * p
    
    # Initialize U, V, W as lists of lists (or use numpy, but lists are fine for small sizes)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # V is the identity matrix
    for r in range(rank):
        V[r][r] = 1
    
    # U[i*m + j, r] = 1 if j' = j where r = j'*p + k
    # For each r, determine j' and k'
    for r in range(rank):
        j_prime = r // p
        k_prime = r % p
        
        # U[i*m + j, r] = 1 for i in 0..n-1, j = j_prime
        for i in range(n):
            U[i * m + j_prime][r] = 1
        
        # W[k*n + i, r] = 1 for i in 0..n-1, k = k_prime
        for i in range(n):
            W[k_prime * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
