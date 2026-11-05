"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    We decompose the 3x3x5 tensor into a direct sum of five 3x3x1 tensors.
    For the 3x3x1 case, the multiplication tensor M_{3,3,1} is equivalent to computing
    the products of 3 rows of A with a fixed 3-vector (column of B).
    
    Actually, let's think more carefully. The tensor T_{i,j,k} = sum_{a,b} A[i,a] * B[a,b] * delta_{b,k}
    for a 3x3 * 3x5 multiplication.
    
    Wait, the tensor for matrix multiplication M_{n,m,p} has entries T_{i,j,k} where
    C[i,j] = sum_k A[i,k] * B[k,j].
    
    For 3x3 * 3x5: C is 3x5, A is 3x3, B is 3x5.
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    So the tensor has indices (i, j, k) where i in [0,2], j in [0,2], k in [0,4].
    T_{i,j,k} = A[i,j] * B[j,k]
    
    We want to write T as sum_{r=0}^{R-1} u_r * v_r * w_r where
    u_r is a vector of size 3*3=9 (flattened A indices)
    v_r is a vector of size 3*5=15 (flattened B indices)
    w_r is a vector of size 5*3=15 (flattened C indices, in W[k*n+i, r] convention)
    
    Actually, the interface says:
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (15, rank)
    W: (p*n, rank) = (15, rank)
    
    And the decomposition is: T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    where the flattened indices are:
    - U index: i*m + j = i*3 + j
    - V index: j*p + k = j*5 + k
    - W index: k*n + i = k*3 + i
    
    The tensor entry T_{i,j,k} = U[i*3+j, r] * V[j*5+k, r] * W[k*3+i, r] summed over r.
    
    So we need: sum_r U[i*3+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1 for all i,j,k,
    where the "1" represents the identity in the sense that the tensor is the matrix multiplication tensor.
    
    Actually no. The matrix multiplication tensor T_{i,j,k} = 1 if and only if the indices correspond
    to a valid multiplication path. Let me re-read.
    
    The standard matrix multiplication tensor M_{n,m,p} has entries:
    M_{i,j,k} = 1 if the (i,j) entry of A times the (j,k) entry of B contributes to (i,k) entry of C.
    
    So M_{i,j,k} = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    And the tensor is the sum over all such (i,j,k) of e_i^A * e_j^B * e_k^C... no wait.
    
    Let me be precise. The bilinear form for matrix multiplication is:
    C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T_{i,j,k} such that the multiplication can be written as:
    C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    And we need: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = delta_{j is the middle index}... 
    
    No, that's not right either. Let me think again.
    
    The matrix multiplication tensor M_{n,m,p} is defined by:
    M = sum_{i,j,k} e_i^* * e_j * e_k^*
    where e_i^* are basis covectors for A, e_j for B, e_k^* for C.
    
    More concretely, if we flatten A into a vector of length n*m, B into length m*p, C into length p*n,
    then the tensor M is a 3-tensor of dimensions (n*m, m*p, p*n) with entries:
    M[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k; else 0.
    
    Wait, that's not quite right either. Let me use the standard definition.
    
    For multiplying an n x m matrix A by an m x p matrix B to get an n x p matrix C:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    The tensor T has dimensions (n*m, m*p, n*p) [or (p*n) depending on convention].
    T is such that: sum_{a,b} T[a,b,c] * A_flat[a] * B_flat[b] = C_flat[c]
    
    No, the tensor decomposition is: T = sum_r u_r * v_r * w_r where
    C_flat[c] = sum_r (sum_a u_r[a] * A_flat[a]) * (sum_b v_r[b] * B_flat[b]) * w_r[c]
    
    So T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    
    And we need: sum_{a,b} T[a,b,c] * A_flat[a] * B_flat[b] = C_flat[c]
    
    Which means: sum_{a,b} sum_r u_r[a] * v_r[b] * w_r[c] * A_flat[a] * B_flat[b] = C_flat[c]
    
    For this to equal C[i,k] = sum_j A[i,j] * B[j,k] for c = k*n+i (or whatever convention):
    
    Let's use the convention from the problem: W index is k*n + i.
    
    So C_flat[k*n + i] = C[i,k] = sum_j A[i,j] * B[j,k]
    
    A_flat[i*m + j] = A[i,j]
    B_flat[j*p + k] = B[j,k]
    
    So we need: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] summed appropriately.
    
    Actually, the bilinear map is:
    C_flat[c] = sum_{a,b} T[a,b,c] * A_flat[a] * B_flat[b]
    
    For c = k*n + i:
    C[i,k] = sum_{a,b} T[a,b,k*n+i] * A_flat[a] * B_flat[b]
    
    We want this to equal sum_j A[i,j] * B[j,k].
    
    So T[a,b,k*n+i] must be such that sum_{a,b} T[a,b,k*n+i] * A_flat[a] * B_flat[b] = sum_j A[i,j] * B[j,k].
    
    This means T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    So the tensor T has entries T[a,b,c] = 1 iff there exist i,j,k such that a = i*m+j, b = j*p+k, c = k*n+i.
    
    Now, for the schoolbook decomposition, we have one term for each (i,j,k):
    u_{i,j,k}[a] = 1 if a = i*m+j else 0
    v_{i,j,k}[b] = 1 if b = j*p+k else 0
    w_{i,j,k}[c] = 1 if c = k*n+i else 0
    
    This gives rank = n*m*p = 3*3*5 = 45.
    
    Now, the directive says to exploit linearity in the third index (k) by decomposing into five 3x3x1 tensors.
    
    For a fixed k, we have a 3x3x1 tensor: T_k[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i.
    
    For fixed k, b = j*5+k, so b ranges over {k, 5+k, 10+k}.
    c = k*3+i, so c ranges over {3k, 3k+1, 3k+2}.
    
    The 3x3x1 tensor for fixed k is essentially the same as a 3x3x1 matrix multiplication tensor,
    which is just the identity-like tensor for multiplying a 3x3 matrix by a 3x1 vector.
    
    For 3x3x1, the schoolbook rank is 9.
    
    But can we do better? The 3x3x1 tensor is equivalent to computing 3 dot products of 3-vectors.
    Each dot product has rank 3 in the bilinear sense... no wait.
    
    Actually, for 3x3x1: C[i,0] = sum_j A[i,j] * B[j,0]
    This is just 3 independent dot products, each of length 3.
    
    Each dot product of two 3-vectors has bilinear rank 3 (you need 3 multiplications).
    So 3 dot products need 3*3 = 9 multiplications.
    
    So the 3x3x1 tensor has rank 9.
    
    Therefore, 5 such tensors give total rank 5*9 = 45, which is the same as schoolbook.
    
    So this approach doesn't improve on schoolbook.
    
    The directive mentions "rank-9 Strassen decomposition for the 3x3x3 block" - but Strassen is for 2x2x2, not 3x3x3.
    And "rank-5 schoolbook decomposition for the remaining 3x3x2 block" - 3x3x2 schoolbook is 18, not 5.
    
    I think the directive contains errors. Let me just implement the schoolbook decomposition correctly,
    which gives rank 45.
    
    Actually, wait. Let me reconsider. Maybe there's a way to do better than 45 for 3x3x5.
    
    The known lower bound for 3x3 matrix multiplication is 19 (Alon-Boppana), and the best known upper bound is 23.
    For 3x3x5, I don't know the exact best rank, but it should be at most 45 and possibly less.
    
    Since I'm not sure of a better decomposition, let me just implement the correct schoolbook decomposition
    which gives rank 45. This is the baseline.
    
    Let me verify the index conventions:
    - U: (n*m, rank) = (9, 45), U[i*3+j, col]
    - V: (m*p, rank) = (15, 45), V[j*5+k, col]
    - W: (p*n, rank) = (15, 45), W[k*3+i, col]
    
    For each (i,j,k), we assign a column index col.
    U[i*3+j, col] = 1
    V[j*5+k, col] = 1
    W[k*3+i, col] = 1
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = n * m * p  # 45
    U = [[0] * rank for _ in range(n * m)]  # 9 x 45
    V = [[0] * rank for _ in range(m * p)]  # 15 x 45
    W = [[0] * rank for _ in range(p * n)]  # 15 x 45
    
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
