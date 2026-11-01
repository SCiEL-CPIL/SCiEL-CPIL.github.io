"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition.
    
    We decompose the multiplication into two independent 1x2 * 2x14 vector-matrix
    products (one for each output row). Each such product has a natural rank of
    m*p = 2*14 = 28 using the schoolbook approach for vector-matrix products.
    
    For a 1xm * mxp product: c_k = sum_j a_j * b_{j,k}
    The tensor has entries T[j, (j',k), k'] where we need:
    T[j, j'p+k, k'n+0] = delta_{j,j'} * delta_{k,k'}
    
    The schoolbook decomposition for 1xm * mxp uses rank m*p:
    For each (j,k), one term: U[j] = 1, V[j*p+k] = 1, W[k*n] = 1
    
    For our 2x2 * 2x14 problem, we have two such independent sub-problems
    (one for i=0, one for i=1), each with rank 2*14 = 28.
    Total rank = 2 * 28 = 56.
    
    However, we can potentially do better by sharing structure. Let's analyze
    more carefully.
    
    The key insight from the directive: exploit the linear structure by
    decomposing into two independent 1x2*2x14 products. But actually, each
    1x2*2x14 product already requires rank 28 in the tensor sense.
    
    Wait - let me reconsider. For 1xm * mxp, the tensor is:
    T in C^{1*m} ⊗ C^{m*p} ⊗ C^{p*1}
    T[(i,j), (j',k), (k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    For 1x2 * 2x14: T has shape (2, 28, 14)
    T[j, j'p+k, k'] = delta_{j,j'} * delta_{k,k'}
    
    The rank-1 decomposition: for each (j,k), we have:
    U[j, r] = 1, V[j*p+k, r] = 1, W[k, r] = 1
    This gives rank = 2*14 = 28.
    
    Can we do better for 1x2 * 2x14? This is equivalent to computing:
    c_k = a_0 * b_{0,k} + a_1 * b_{1,k} for k = 0,...,13
    
    This is a bilinear map from C^2 x C^{28} to C^{14}.
    The tensor rank of this is at least max(2, 28, 14) = 28? No, that's not right.
    
    Actually, for the tensor T_{j, (j',k), k'} = delta_{j,j'} delta_{k,k'},
    this is the identity tensor in some sense. The rank is exactly m*p = 28.
    
    So for our problem, the total rank is 2 * 28 = 56, which is the same as
    schoolbook. There's no improvement possible with this decomposition.
    
    Let me just return the schoolbook decomposition with rank 56.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
