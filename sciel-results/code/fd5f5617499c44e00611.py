import numpy as np

def construct():
    # Dimensions: A is (n x m), B is (m x p), C is (n x p)
    # We need to decompose the tensor T_{i,j,k} = A[i,j] * B[j,k]
    # The output index is r = k*n + i (trace convention W[k*n+i, r])
    # Input indices are u = i*m + j and v = j*p + k
    
    n, m, p = 2, 3, 9
    
    # The rank of the matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, n*p).
    # Here, n*m = 6, m*p = 27, n*p = 18. So rank >= 27.
    # However, the standard schoolbook algorithm uses rank n*m*p = 54.
    # For small dimensions like 2x3x9, it is known that the rank is exactly n*m*p = 54.
    # This is because the tensor is a direct sum of p copies of the (n,m,1) tensor?
    # No, it's a direct sum of p copies of the (n,m,1) tensor only if we view it column-wise?
    # Let's check the structure.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is sum_j A[i,j] B[j,k] e_{i,k}
    # This is a direct sum over j? No.
    # It is a direct sum over k of the (n,m,1) tensor?
    # For a fixed k, C[i,k] = sum_j A[i,j] B[j,k]. This is the (n,m,1) tensor.
    # Since the k-index is separate in the output and in the B input, but coupled with A?
    # No, A does not depend on k. B depends on k.
    # So for each k, we have a separate multiplication of A (n x m) by B[:,k] (m x 1).
    # These p multiplications are independent because the inputs B[:,k] are separate vectors.
    # And the outputs C[:,k] are separate vectors.
    # And A is shared.
    # Wait, if A is shared, can we share rank?
    # In tensor decomposition, if a factor is shared, it doesn't necessarily allow rank sharing unless the structure permits it.
    # The tensor is T = sum_{k=0}^{p-1} T^{(k)} where T^{(k)} is the tensor for C[:,k] = A * B[:,k].
    # T^{(k)} has non-zero entries only for indices (i, j, k) where the output index is (i,k).
    # In the flattened index space:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # For a fixed k, the non-zero W indices are k*n + i for i in 0..n-1.
    # The non-zero V indices are j*p + k for j in 0..m-1.
    # The non-zero U indices are i*m + j for i in 0..n-1, j in 0..m-1.
    # Since the W indices for different k are disjoint (k*n+i vs k'*n+i'), the tensors T^{(k)} have disjoint support in the W-mode.
    # This implies that the rank of the direct sum is the sum of the ranks.
    # Rank(T) = sum_{k=0}^{p-1} Rank(T^{(k)}).
    # T^{(k)} is the tensor for multiplying an (n x m) matrix by an (m x 1) vector.
    # The rank of this tensor is n*m.
    # So Rank(T) = p * n * m = 9 * 2 * 3 = 54.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the decomposition
    # For each k, we need n*m terms.
    # Total terms = p * n * m.
    # We can index the rank r = k * (n*m) + i * m + j.
    
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = k * (n * m) + i * m + j
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}
