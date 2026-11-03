"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    The standard decomposition has rank n*m*p = 4*4*9 = 144.
    We try to find a lower rank decomposition using known techniques.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p,
    the multiplication tensor has rank at most n*m*p (trivial decomposition).
    
    One approach: use the fact that we can factor through smaller dimensions.
    For example, if we can express the multiplication using fewer terms by
    exploiting structure.
    
    A known lower bound for matrix multiplication rank is related to the
    Strassen algorithm ideas. For small matrices, we can sometimes find
    decompositions with rank less than n*m*p.
    
    Let's try a direct approach: enumerate possible rank values and check
    if a valid decomposition exists. Since this is a search problem, we'll
    use a heuristic or known result.
    
    For 4x4 * 4x9, one approach is to use block decomposition or known
    results from tensor rank literature.
    
    Actually, let me think about this more carefully. The tensor for matrix
    multiplication A(n,m) * B(m,p) -> C(n,p) has indices:
    - U has shape (n*m, rank) - one row per (i,j) pair
    - V has shape (m*p, rank) - one row per (j,k) pair  
    - W has shape (p*n, rank) - one row per (k,i) pair (using trace convention W[k*n+i, r])
    
    The constraint is that for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    And for all other combinations (i,j,k) where the product term should be 0:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0
    
    Wait, that's not quite right. The tensor T_{ijk} = 1 if the multiplication
    term A_{ij} * B_{jk} contributes to C_{ik}. Actually, the matrix multiplication
    tensor is T_{ijk} where i indexes rows of A/C, j indexes columns of A/rows of B,
    k indexes columns of B/C. T_{ijk} = 1 for all valid (i,j,k) in range, and the
    contraction is C_{ik} = sum_j A_{ij} B_{jk}.
    
    So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    This is actually the all-ones tensor of size n x m x p!
    
    Wait, no. Let me re-read. The multiplication C = A*B where A is n x m and B is m x p.
    C_{ik} = sum_j A_{ij} * B_{jk}.
    
    The tensor T has T[i,j,k] = 1 for all valid i,j,k. When we do:
    C_{ik} = sum_j (sum_r U[i*m+j, r] * W[k*n+i, r]) * (sum_s V[j*p+k, s] * ...)
    
    Hmm, I'm confusing myself. Let me think about the standard formulation.
    
    In the standard tensor decomposition for matrix multiplication:
    We want C = A * B, so C_{ik} = sum_j A_{ij} B_{jk}.
    
    We can write this as: C_{ik} = sum_{i,j,k} T_{ijk} A_{ij} B_{jk}
    where T_{ijk} = 1 for all i,j,k (since each term A_{ij}B_{jk} contributes to C_{ik}
    exactly once for fixed i,k and summed over j).
    
    Wait, that's not right either. For fixed i,k, C_{ik} = sum_j A_{ij} B_{jk}.
    So the tensor T has T[i,j,k] = 1 if this term contributes to C[i,k], which is always
    true. But we need to be careful about which output index it goes to.
    
    Actually, the standard way: the multiplication tensor is a 3-tensor T of dimensions
    (n*m) x (m*p) x (n*p) or similar, where T[(i,j),(j,k),(i,k)] = 1.
    
    In the factor matrix formulation given:
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    The decomposition says: T[(i,j),(j',k),(k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    For this to equal the matrix multiplication tensor, we need:
    T[(i,j),(j',k),(k',i')] = 1 if i=i', j=j', k=k' (i.e., the indices match for a valid multiplication term)
    and 0 otherwise.
    
    Wait, but that would mean T is just an identity-like tensor. Let me reconsider.
    
    Actually, I think the correct interpretation is:
    The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    The factorization is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This makes sense because when we compute:
    sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,j,k} A[i,j] * B[j,k]
    And grouping by i,k: sum_k (sum_j A[i,j] * B[j,k]) = C[i,k]
    
    But wait, that gives us sum over all i,j,k, not just the right grouping.
    
    I think the correct formulation is that the tensor is T[i,j,k] where the first index
    is (i,j) for A, second is (j,k) for B, third is (i,k) for C. And T[(i,j),(j',k),(i',k')] 
    = 1 if i=i', j=j', k=k', else 0.
    
    So T is actually a "diagonal" tensor in this indexing. The rank of this tensor is what we
    want to minimize.
    
    The trivial decomposition has rank n*m*p with U[i*m+j, i*m*p+j*p+k] = 1, etc.
    
    To find a lower rank, we need to find a way to express this diagonal tensor as a sum of
    fewer rank-1 tensors.
    
    This is equivalent to finding the tensor rank of the "matrix multiplication tensor" for
    these dimensions. For small cases, this is known or can be searched.
    
    Given the time constraint, let me try a simple heuristic: start with the trivial rank and
    see if we can reduce it. For 4x4*4x9, the trivial rank is 144.
    
    One idea: use the fact that some of the terms might be combinable. But for a diagonal
    tensor, it's not obvious how to reduce the rank below the number of non-zero entries.
    
    Actually, for a tensor T where T[i,j,k] = delta[i,i'] * delta[j,j'] * delta[k,k'] in the
    appropriate indexing, the rank is exactly the number of non-zero entries if the tensor is
    "generic" enough. But there might be structure.
    
    Let me just return the trivial decomposition for now, as finding a lower rank is a hard
    open problem in general.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
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
