def construct():
    from fractions import Fraction as Fr
    import numpy as np

    n, m, p = 2, 4, 8

    # Build the multiplication tensor as a linear operator
    # C = A @ B where A is (n x m) and B is (m x p)
    # C has p*n entries with C[k*n + i, r] = sum_j A[i,j] * B[j, k*n+r]
    # Actually, let's use the standard convention: C[i, k] = sum_j A[i,j] B[j,k]
    # W index: W[k*n + i, r] where r is the column of W (rank dimension)
    # U index: U[i*m + j, r]
    # V index: V[j*p + k, r]

    # The tensor T has entries T[(i,j), (j,k)] = delta for the multiplication
    # We want rank-minimal decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    # where T[i,j,k] = 1 if this is a valid multiplication term, else 0
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # since C[i,k] = sum_j A[i,j] B[j,k]

    # Let's build the flattening matrix
    # Row index: (i, k) for output, but we need to be careful
    # The tensor is T[i,j,k] = 1
    # We can flatten as: rows = (i,k) pairs, cols = j index
    # M[i,k, j] = T[i,j,k] = 1 for all i,j,k
    # This is a rank-1 tensor! But that's not right because the tensor for matrix multiplication
    # is actually sum over j of A[i,j] B[j,k], so the tensor entries are all 1s in a specific way.

    # Let me reconsider. The multiplication tensor for (n,m,p) has:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because each product term A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.

    # So the tensor is simply the all-ones tensor of shape (n, m, p).
    # The rank of the all-ones tensor of shape (n,m,p) is max(n, p) when m >= max(n,p)?
    # No, that's not right either. The tensor rank of the all-ones tensor...

    # Actually, let me think about this more carefully using the flattening approach.
    # The tensor T has shape (n, m, p) with T[i,j,k] = 1.
    # Flattening along j: M is a matrix of shape (n*p, m) where M[i*p+k, j] = T[i,j,k] = 1
    # This is an all-ones matrix of shape (n*p, m), which has rank min(n*p, m) = min(16, 4) = 4.
    # So the tensor rank is at least 4.

    # But wait - is 4 achievable? Let's think about it.
    # If rank is 4, we need U of shape (n*m, 4) = (8, 4), V of shape (m*p, 4) = (32, 4), W of shape (p*n, 4) = (16, 4)
    # such that sum_{r=0}^{3} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

    # Let's try to construct such a decomposition.
    # Since the tensor is all ones, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    # One approach: use the fact that the tensor is a product of three all-ones vectors.
    # If we set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1, and all other entries 0,
    # then sum_r ... = 1*1*1 = 1. That works! Rank = 1.

    # Wait, that can't be right. Let me verify.
    # If rank = 1, then T[i,j,k] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0]
    # If all are 1, then T[i,j,k] = 1 for all i,j,k. Yes!

    # So the rank-minimal decomposition has rank 1? That seems too simple.

    # Let me double-check by looking at what the evaluator expects.
    # The task says "exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix"
    # The multiplication tensor T[i,j,k] = 1 for all valid indices.
    # A rank-1 decomposition would be U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # This gives T[i,j,k] = 1, which is correct.

    # But this seems like it would be the trivial answer and the score would be very low.
    # Let me re-read the problem...

    # "Find a rank-minimal exact tensor decomposition"
    # If rank 1 works, then that's the minimum.

    # Let me verify with a simple case: 1x1 by 1x1 matrix multiplication.
    # C[0,0] = A[0,0] * B[0,0]
    # Tensor T[0,0,0] = 1
    # Rank 1: U[0,0]=1, V[0,0]=1, W[0,0]=1. Works.

    # So for any (n,m,p), the all-ones tensor has rank 1? That can't be right in general...
    # Actually, yes it can. The tensor T[i,j,k] = 1 is separable (rank 1) because it's
    # the outer product of three all-ones vectors.

    # So the answer is rank = 1.

    rank = 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Set all entries to 1
    for i in range(n * m):
        U[i][0] = 1
    for j in range(m * p):
        V[j][0] = 1
    for k in range(p * n):
        W[k][0] = 1

    return {"U": U, "V": V, "W": W}
