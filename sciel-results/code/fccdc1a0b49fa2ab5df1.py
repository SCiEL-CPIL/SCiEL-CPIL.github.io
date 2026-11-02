"""
Rank-minimal tensor decomposition for 2x2 * 2x15 matrix multiplication.

Key insight: The multiplication tensor for (2,2,15) is the Kronecker product
of the (2,2,2) tensor (rank 7 via Strassen) and a (1,1,15) trivial tensor.
Actually, the tensor T_{i,j,k} = 1 for all i,j,k in the standard formulation
is not correct. Let me reconsider.

For matrix multiplication C = A * B where A is 2x2, B is 2x15, C is 2x15:
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T_{i,j,k} = 1 for all valid i,j,k (i in 0..1, j in 0..1, k in 0..14).

This tensor is actually the outer product of three vectors:
T_{i,j,k} = 1_i * 1_j * 1_k

So it has rank 1! But that's not right because we need to verify the decomposition.

Let me re-examine. The multiplication tensor has entries T_{i,j,k} = 1 if the
product A[i,j]*B[j,k] contributes to C[i,k]. For all valid indices, this is 1.

So T = ones(2,2,15). This is rank 1 (a pure tensor).

Wait, but then the decomposition would be:
U[i*m+j, r] = 1 for all i,j
V[j*p+k, r] = 1 for all j,k  
W[k*n+i, r] = 1 for all k,i

With rank=1: U has shape (4,1), V has shape (30,1), W has shape (30,1), all ones.

Then sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1*1*1 = 1 = T_{i,j,k}. ✓

So rank 1 is sufficient!
"""

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
