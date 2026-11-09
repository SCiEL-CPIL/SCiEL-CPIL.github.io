import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _rank19_1x5x5():
    """
    Returns a rank-19 decomposition for 1x5x5 matrix multiplication.
    This is based on the known result that the tensor rank of 1x5x5 is 19.
    We use a construction from literature or derive one.
    
    Actually, let's use a simpler approach: we know that for 1xN x NxN,
    the rank is at most N^2 (schoolbook) but can be lower.
    For 1x5x5, a rank-19 decomposition exists.
    
    Let me construct one using the following known method:
    The tensor T_{i,j,k} = delta_{i,0} * (A_{j,k}) where A is 5x5.
    Wait, the tensor for 1x5x5 multiplication is:
    T_{i,j,k} where i in 1..1, j in 1..5, k in 1..5
    The multiplication tensor is sum_{i,j,k} T_{i,j,k} * e_i ⊗ e_j ⊗ e_k
    where T_{i,j,k} = 1 if the output entry (i, k) depends on input entries...
    
    Actually, for matrix multiplication C = A * B where A is 1x5 and B is 5x5:
    C_{1,k} = sum_{j=1}^{5} A_{1,j} * B_{j,k}
    
    So the tensor is: sum_{j=1}^{5} sum_{k=1}^{5} A_{1,j} * B_{j,k} * C_{1,k}
    
    The standard tensor decomposition has rank 25 (one term per j,k pair).
    
    For rank 19, we need a more clever decomposition. This is non-trivial.
    
    Let me look for a known rank-19 decomposition for 1x5x5.
    
    Actually, I recall that the rank of 1xn x nxn matrix multiplication is n^2 for n <= 4,
    but for n=5 it might be less. Let me check: the rank of the 1x5x5 tensor is actually 25?
    
    Wait, I need to be careful. The tensor rank of the matrix multiplication tensor
    for 1x5x5 might not be 19. Let me reconsider.
    
    Actually, for 1xn x nxn, the tensor is:
    T = sum_{j=1}^n sum_{k=1}^n e_{1} ⊗ e_j ⊗ e_k
    
    This tensor has rank n^2 because each term e_1 ⊗ e_j ⊗ e_k is linearly independent
    in the tensor space. So the rank is exactly 25 for 1x5x5.
    
    Hmm, so my inspiration was wrong. The rank of 1x5x5 is 25, not 19.
    
    Let me reconsider the problem. The total tensor for 3x5x5 is:
    T = sum_{i=1}^3 sum_{j=1}^5 sum_{k=1}^5 e_i ⊗ e_j ⊗ e_k
    
    This is a direct sum of three 1x5x5 tensors (one for each i).
    The rank of a direct sum is at most the sum of the ranks, and in this case
    since the tensors have disjoint supports in the first index, the rank is exactly
    3 * rank(1x5x5) = 3 * 25 = 75.
    
    So the schoolbook rank of 75 is actually minimal for this structure?
    
    Wait, no. The tensor for 3x5x5 matrix multiplication is:
    T = sum_{i=1}^3 sum_{j=1}^5 sum_{k=1}^5 e_i ⊗ e_j ⊗ e_k
    
    But this can potentially be decomposed more efficiently if we exploit
    structure across the different i values.
    
    Actually, the tensor is:
    T = sum_{j=1}^5 sum_{k=1}^5 (sum_{i=1}^3 e_i ⊗ e_j ⊗ e_k)
    
    Each term (sum_{i=1}^3 e_i) ⊗ e_j ⊗ e_k is a rank-3 tensor in the first factor.
    
    The question is whether we can find a decomposition of rank less than 75.
    
    For now, let me just keep the schoolbook decomposition since I can't easily
    find a better one. The rank is 75.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    # For 3x5x5, the schoolbook rank is 75
    # The tensor is a direct sum of three 1x5x5 tensors, each of rank 25
    # So total rank is 75
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
