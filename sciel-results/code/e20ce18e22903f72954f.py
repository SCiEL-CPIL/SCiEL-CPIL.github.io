"""
Tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: Use the known rank-14 decomposition for 4x4 matrix multiplication
(achieved by Strassen-like or Winograd-like methods) along the n and m dimensions,
combined with schoolbook along the p=8 dimension. This gives rank 14*14*8 = 1568,
but that's worse than schoolbook (128).

Actually, let's reconsider. Schoolbook for 4x4*4x8 is rank 4*4*8 = 128.

The best known for square 4x4 is rank 14 (from literature on small matrix multiplication).
For 4x4*4x8, we can think of it as 8 separate 4x4*4x1 multiplications.
Or as 4 separate 4x4*4x2 multiplications.

Let me use the fact that the tensor for (n,m,p) = (4,4,8) multiplication has
rank at most the rank of (4,4,4) times the rank of (4,4,4) or similar decompositions.

Actually, a cleaner approach: use the rank-14 decomposition for 4x4 matrix mult.
The 4x4*4x8 product can be seen as: for each output column k (8 columns),
C[:,k] = A * B[:,k], which is 4x4 * 4x1.

So we need 8 separate 4x4*4x1 products. If we use a rank-14 decomposition for 4x4*4x4,
each 4x4*4x1 uses rank 14 (by fixing one column), so total rank = 8*14 = 112.

Wait, that's not quite right. Let me think more carefully.

The tensor T for (n,m,p) matrix multiplication has entries T[i,j,k] = sum over appropriate indices.
For (4,4,8): T[i,j,k] where i in [0,4), j in [0,4), k in [0,8).

The decomposition rank R means we can write:
C[i,k] = sum_{r=0}^{R-1} U[i, r] * V[j, r] * W[k, r] ... no wait.

Actually the standard formulation: (A * B)[i,k] = sum_j A[i,j] * B[j,k]

The tensor decomposition uses:
(A * B)[i,k] = sum_r U[i, r] * (sum_j U2[j, r] * A[i,j]) * (sum_k ... )

Let me use the standard format from the problem:
U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)

And the decomposition is:
(A*B)[i,k] = sum_r sum_j sum_l  U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r] * A[i,j] * B[j,l]... 

Hmm, let me re-read the interface. The standard bilinear algorithm:
C = A*B where A is n x m, B is m x p, C is n x p.

The tensor T has T[i, j, k] for the coefficient of A[i,j]*B[j,k] in C[i,k].
But actually it's more nuanced. The multiplication tensor has rank equal to the minimum number of bilinear forms.

The interface says: U[n*m, rank], V[m*p, rank], W[p*n, rank]

And the decomposition should satisfy:
For all i in [0,n), k in [0,p):
  C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]

The bilinear decomposition uses:
C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+l, r] * B[j,l])... 

No, I think the standard is:
Let's define for each rank element r:
  a_r = sum_{i,j} U[i*m+j, r] * A[i,j]
  b_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (but this doesn't seem right dimensionally)

Actually, I think the correct interpretation is:
The multiplication tensor T[i,j,k] (where the product involves A[i,j] and B[j,k] contributing to C[i,k])
can be decomposed as T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait, that would mean:
sum_j,k A[i,j]*B[j,k]*T[i,j,k] = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * (sum_{j,k} V[j*p+k,r]*B[j,k]) * ... 

No, that's not a bilinear decomposition.

Let me think about this differently. The standard bilinear algorithm for matrix multiplication:
C = A * B

We express each entry C[i,k] as:
C[i,k] = sum_r M_r * N_r

where M_r = sum_{i,j} alpha_r[i,j] * A[i,j] and N_r = sum_{j,k} beta_r[j,k] * B[j,k]

But that's not quite right either because M_r and N_r need to combine to give C[i,k] specifically.

The correct formulation: For each pair (i,k), we have:
C[i,k] = sum_j A[i,j] * B[j,k]

In a bilinear decomposition of rank R:
C[i,k] = sum_{r=0}^{R-1} (sum_j U_r[i,j] * A[i,j]) * (sum_l V_r[j,l] * B[j,l]) * W_r[i,k]

Wait, I think the W matrix handles the combination. Let me just use the schoolbook as a baseline and try to find a better decomposition.

Schoolbook: rank = n*m*p = 4*4*8 = 128.

For 4x4 matrix multiplication, the best known rank is 14 (from the literature on small matrix multiplication, e.g., using the Strassen algorithm adapted for 4x4, or specialized algorithms).

If we can achieve rank 14 for 4x4*4x4, then for 4x4*4x8 we might achieve rank 14*8/something... 

Actually, here's a cleaner approach: The 4x4*4x8 multiplication can be viewed as computing 8 output columns, each being a 4x4*4x1 product. If we have a rank-R decomposition for 4x4*4x1 (matrix-vector product), then total rank is 8*R.

For 4x4*4x1, the schoolbook rank is 16. Can we do better? The matrix-vector product A*x where A is 4x4 and x is 4x1. This is 16 scalar multiplications in schoolbook. It's not clear we can reduce this significantly without changing the structure.

Alternatively, think of it as 2 blocks: A is 4x4, B is 4x8 = two 4x4 blocks. 
C = A * [B1 | B2] = [A*B1 | A*B2]

If we use rank-14 for each 4x4*4x4, total rank = 2*14 = 28? No, that's not how it works.

Let me just implement the schoolbook (rank 128) and verify it works, then try to optimize.

Actually, I recall that for matrix multiplication of size (n, m, p), the rank is at most n*m*p (schoolbook), and for square n=n=n, the best known is less than n^3. For 4x4, rank 14 is known.

Let me implement the schoolbook first to ensure correctness, then attempt to use known 4x4 decompositions.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 8]


def construct():
    n, m, p = DIMENSIONS
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
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}")
    print(f"V shape: {V.shape}")
    print(f"W shape: {W.shape}")
