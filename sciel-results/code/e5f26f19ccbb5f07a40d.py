"""
Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

Approach:
  - The tensor T for C = A B with A in R^{2x4}, B in R^{4x5}, C in R^{2x5}
    has T[i, j, k] = 1 for all i, j, k (each output C[i,k] is the sum over
    j of A[i,j]*B[j,k]).
  - A rank-1 term (U, V, W) with U[i,j,c] = u_i, V[j,k,c] = v_k, W[k,i,c] = w_i
    contributes u_i * v_k * w_i to T[i,j,k] for every (i,j,k).
  - We need sum_c u_i^{(c)} v_k^{(c)} w_i^{(c)} = 1 for all (i,k).
  - This is a system of n*p = 10 equations in 3*r unknowns.
  - For r = 10 we can set up a linear system in the products z_{i,k}^{(c)} = u_i^{(c)} w_i^{(c)} * v_k^{(c)}.
  - Actually, let's use a simpler construction.

  Key insight: T[i,j,k] = 1 for all i,j,k.
  If we set U[i*m+j, c] = a_{i,c}, V[j*p+k, c] = b_{k,c}, W[k*n+i, c] = c_{i,c},
  then T[i,j,k] = sum_c a_{i,c} * b_{k,c} * c_{i,c}.
  We need sum_c (a_{i,c} * c_{i,c}) * b_{k,c} = 1 for all i,k.

  Let d_{i,c} = a_{i,c} * c_{i,c}. Then we need:
    sum_c d_{i,c} * b_{k,c} = 1 for all i, k in {0,1} x {0,1,2,3,4}.

  This means the matrix D (n x r) times B^T (r x p) = J (n x p all-ones).
  So D * B^T = J.

  We can choose D and B^T to be any factorization of J.
  The simplest: D = ones(n, 1) (rank 1), B^T = ones(1, p).
  Then r = 1, and a_{i,0} * c_{i,0} = 1, b_{k,0} = 1.
  Set a_{i,0} = 1, c_{i,0} = 1, b_{k,0} = 1.
  Then U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
  Check: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
              = 1 * 1 * 1 = 1. Correct!

  So rank 1 works!

  Wait, but the seed program uses rank 40 (schoolbook). Let me re-examine.
  The schoolbook decomposition has one term per (i,j,k) triple, giving rank 40.
  But the tensor T[i,j,k] = 1 for all i,j,k is indeed rank 1.

  Let me verify: with rank 1,
  U has shape (n*m, 1) = (8, 1), all ones.
  V has shape (m*p, 1) = (20, 1), all ones.
  W has shape (p*n, 1) = (10, 1), all ones.
  T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1*1*1 = 1.
  Yes, this is correct.

  So the minimal rank is 1.
"""
import numpy as np

DIMENSIONS = [2, 4, 5]


def construct():
    """
    Returns a rank-1 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    The multiplication tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]
    and each term A[i,j]*B[j,k] contributes 1 to the tensor).
    
    A rank-1 decomposition exists:
      U[i*m+j, 0] = 1 for all i,j
      V[j*p+k, 0] = 1 for all j,k
      W[k*n+i, 0] = 1 for all k,i
    
    This gives T[i,j,k] = 1 * 1 * 1 = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 1
    
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
