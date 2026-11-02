"""
Find a rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication.

Strategy:
1. Use the known schoolbook decomposition as a fallback (rank 90).
2. Attempt to find a lower-rank decomposition using random linear algebra /
   structured search over the tensor space.
3. The key insight: for matrix multiplication, the tensor T has components
   T[i,j,k] = 1 if the (i,k)-th entry of the product depends on A[i,j]*B[j,k].
   For 3x5 * 5x6, T is a 3x5x6 tensor where T[i,j,k]=1 for all i,j,k (since
   every product entry sums over all j).

   Actually, for standard matrix multiplication C[i,k] = sum_j A[i,j] B[j,k],
   the multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m),
   k in [0,p). So it's the all-ones tensor of shape (3,5,6).

   Wait, that's not right. The tensor for matrix multiplication is:
   T_{i,j,k} = 1 if and only if the (i,k) entry of the product uses A_{i,j} * B_{j,k}.
   For standard matrix multiplication, C_{i,k} = sum_j A_{i,j} B_{j,k}, so
   every (i,j,k) triple contributes. Thus T is the all-ones tensor of shape (3,5,6).

   The rank of the all-ones tensor of shape (n,m,p) is known to be related to
   the rank of matrix multiplication. For the all-ones tensor, the rank is
   actually the rank of the matrix multiplication tensor, which for (3,5,6)
   is at most 90 (schoolbook).

   However, we can try to find a lower-rank decomposition by exploiting
   symmetries and using random linear combinations.

   A better approach: use the fact that the all-ones tensor can be decomposed
   using the rank of the associated matricizations. The tensor T (shape 3x5x6)
   can be seen as a 3x(5*6) matrix, a 5x(3*6) matrix, or a 6x(3*5) matrix.
   The rank of T is at least the maximum of these matricization ranks.

   The 3x30 matricization has rank at most 3, the 5x18 has rank at most 5,
   and the 6x15 has rank at most 6. So the tensor rank is at least 6.

   In fact, for the all-ones tensor, the rank is exactly max(n, m, p) if we
   allow real entries? No, that's not correct either.

   Let me think again. The all-ones tensor of shape (n,m,p) has tensor rank
   equal to the rank of the matrix multiplication tensor for (n,1,p) * (1,m,p)?
   No.

   Actually, the all-ones tensor T_{i,j,k} = 1 for all i,j,k can be written as:
   T = sum_{r=1}^{R} u_r v_r w_r
   where u_r is in R^n, v_r in R^m, w_r in R^p.

   This is equivalent to saying that the n x m matrix of ones, the m x p matrix
   of ones, etc. can be factored.

   The tensor rank of the all-ones tensor of shape (n,m,p) is exactly 
   max(n, m, p) if we're over a field of characteristic 0? Let me verify for
   small cases.

   For (2,2,2): the all-ones tensor. Can it be written as sum of 2 rank-1 tensors?
   T_{ijk} = 1. We need u_r, v_r, w_r such that sum_r u_{r,i} v_{r,j} w_{r,k} = 1.
   If R=2, we need two triples of vectors. This is equivalent to the matrix
   of ones being decomposable. I believe the rank is actually 2 for (2,2,2).

   For general (n,m,p), I think the rank of the all-ones tensor is max(n,m,p).

   So for (3,5,6), the rank should be 6.

   Let me construct a rank-6 decomposition:
   We need 6 vectors u_r (length 3), v_r (length 5), w_r (length 6) such that
   sum_{r=1}^{6} u_{r,i} v_{r,j} w_{r,k} = 1 for all i,j,k.

   One way: let w_r be the standard basis vectors in R^6, so w_r has a 1 at
   position r and 0 elsewhere. Then we need sum_{r=1}^{6} u_{r,i} v_{r,j} = 1
   for each k... wait, no. If w_r is the r-th standard basis vector, then
   for a fixed k, only r=k contributes, so we need u_{k,i} v_{k,j} = 1 for all i,j
   when k is in range. But we only have 6 values of r, and k goes from 0 to 5.
   For k=0: u_0,i * v_0,j = 1 for all i,j. This means u_0 is a constant vector
   (all 1s) and v_0 is a constant vector (all 1s). Similarly for k=1,...,5.
   But then for k=0, we get 1; for k not in {0,...,5}, we get 0. But k only goes
   from 0 to 5, so this works! Each r corresponds to a k value.

   So: u_r = (1,1,1) for all r, v_r = (1,1,1,1,1) for all r, w_r = e_r (standard
   basis in R^6). Then sum_r u_{r,i} v_{r,j} w_{r,k} = sum_r 1*1*delta_{r,k} = 1
   for each k in {0,...,5}. 

   So the rank-6 decomposition is:
   U[i, r] = 1 for all i, r
   V[j, r] = 1 for all j, r  
   W[k, r] = 1 if k==r else 0

   But wait, the output format requires:
   U: shape (n*m, rank) = (15, 6)
   V: shape (m*p, rank) = (30, 6)
   W: shape (p*n, rank) = (18, 6)

   And the convention is W[k*n+i, r].

   So U[i*m+j, r] = 1 for all i,j,r
   V[j*p+k, r] = 1 for all j,k,r
   W[k*n+i, r] = 1 if k==r else 0... but r goes from 0 to 5, and k goes from 0 to 5.
   So W[k*n+i, r] = delta_{k,r} for all i.

   Let me verify: the product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
   = sum_r 1 * 1 * delta_{k,r} = 1 for each (i,j,k). Correct!

   So the rank is 6, not 90. This is a huge improvement.
"""

import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The all-ones tensor of shape (n,m,p) has rank max(n,m,p) = 6.
    # Construction: R = p = 6
    # U[i*m+j, r] = 1 for all i, j, r
    # V[j*p+k, r] = 1 for all j, k, r
    # W[k*n+i, r] = 1 if k == r else 0
    rank = p  # = 6
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            W[k * n + i][k] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
