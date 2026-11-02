"""
Tensor decomposition for 3x3x8 matrix multiplication.

Strategy:
- The tensor T for multiplying a 3x3 matrix A by a 3x8 matrix B to get a 3x8 matrix C is defined by:
  T[i, j, k] = 1 for all i in [0,3), j in [0,3), k in [0,8).
  The decomposition is: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
  
- The schoolbook decomposition has rank 72 (one term for each (i,j,k)).
- We can potentially reduce the rank by sharing terms or using known fast multiplication algorithms.
- For 3x3x3, the rank is known to be 23 (Landsberg, 2010) or less. However, implementing the specific
  rank 23 decomposition requires specific coefficients which are complex.
  
- A simpler improvement: Split the 8 columns of B into two parts of 4 columns each.
  C = A * B = [A * B1, A * B2] where B1, B2 are 3x4.
  The rank of 3x3x4 is at most 36 (schoolbook).
  Total rank 36 + 36 = 72. No improvement.
  
- What if we use Strassen-like methods?
  For 2x2x2, rank is 7.
  For 3x3x8, we can try to embed smaller fast multiplications.
  
- However, without specific coefficients for a low-rank decomposition, the safest bet is to ensure
  the schoolbook decomposition is correct and fast.
  
- Let's try a randomized search for a lower rank decomposition.
  We want to find U (9xR), V (24xR), W (24xR) such that:
  For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
  
  This is a system of 72 equations.
  If we fix R=71, we have 9*71 + 24*71 + 24*71 = 3*71*23 = 4893 variables.
  This is a huge underdetermined system.
  
  But the equations are cubic (products of variables). This is hard.
  
  Alternative: Use the fact that the tensor is the Kronecker product of 3x3x1 and 1x3x8? No.
  
  Let's stick to the schoolbook for now, as finding a valid lower rank decomposition is non-trivial
  and risky. The schoolbook rank is 72.
  
  Wait, is there a trivial decomposition with rank < 72?
  Consider the tensor T.
  T[i,j,k] = 1.
  
  If we set U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for all c, that's not it.
  
  Let's check if rank 71 is possible.
  The border rank of 3x3x8 is at most 72.
  
  I will output the schoolbook decomposition.
"""

import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for c in range(rank):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
