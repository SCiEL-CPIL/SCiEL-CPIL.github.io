"""
Tensor decomposition for 3x4x6 matrix multiplication.

Approach:
- The matrix multiplication tensor T has entries T[i, j, k; i, k] = 1 (in the flattening sense).
- A rank-r decomposition corresponds to r triples of linear forms (u_r, v_r, w_r) such that:
  sum_j A[i,j] B[j,k] = sum_r (sum_{i',j'} u_r[i',j'] A[i',j']) * (sum_{j',k'} v_r[j',k'] B[j',k']) * w_r[k, i]
  
  Wait, let me re-read the convention. The task says:
  "output factor follows the official trace convention W[k*n+i,r]"
  
  And the decomposition is:
  C[i,k] = sum_j A[i,j] B[j,k]
  
  In the bilinear map formulation:
  C = A * B
  
  We write C as a sum of rank-1 tensors in the space of bilinear maps.
  A rank-1 term is: (linear form on A) * (linear form on B) * (linear form on C^*)
  
  Actually, the standard formulation is:
  The multiplication map mu: R^{n*m} x R^{m*p} -> R^{n*p}
  mu(A, B) = A * B
  
  This is a bilinear map, so it corresponds to a tensor in R^{n*m} * R^{m*p} * R^{n*p}
  
  A rank-1 tensor is u ⊗ v ⊗ w where u in R^{n*m}, v in R^{m*p}, w in R^{n*p}
  The tensor is: T(a, b) = sum_r (u_r^T a) (v_r^T b) w_r
  
  So we need: A * B = sum_r (u_r^T A) (v_r^T B) w_r
  
  In components:
  (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_{i',j'} u_r[i',j'] A[i',j']) * (sum_{j',k'} v_r[j',k'] B[j',k']) * w_r[i,k]
  
  For this to hold for all A, B, we need the tensor decomposition to be exact.
  
  The schoolbook solution uses rank n*m*p = 72, with:
  u_{(i,j,k)}[i',j'] = delta_{i,i'} delta_{j,j'}
  v_{(i,j,k)}[j',k'] = delta_{j,j'} delta_{k,k'}
  w_{(i,j,k)}[i',k'] = delta_{i,i'} delta_{k,k'}
  
  Now, can we do better? For 3x4x6, let's think about block structure.
  
  Actually, let me try a different approach. The key insight for fast matrix multiplication is that
  we can sometimes combine terms. For example, Strassen's algorithm for 2x2 uses 7 multiplications
  instead of 8.
  
  For 3x4x6, one approach is to use the fact that 3x4x6 can be related to other known decompositions.
  
  Let me try to think about this more carefully. The tensor rank of matrix multiplication M_{n,m,p}
  is the minimum number of bilinear multiplications needed.
  
  Known results:
  - M_{2,2,2}: rank 7 (Strassen)
  - M_{3,3,3}: rank <= 23 (Landsberg's bound), unknown if minimal
  
  For M_{3,4,6}, I don't have a specific known lower rank. However, we can try to construct
  a decomposition using block matrix multiplication techniques.
  
  One idea: Split the 3x4 matrix A into blocks and the 4x6 matrix B into blocks, then use
  a known fast algorithm for the resulting block multiplications.
  
  For example, if we split A as [A1; A2] where A1 is 2x4 and A2 is 1x4, and B as [B1 B2]
  where B1 is 4x3 and B2 is 4x3, then:
  C = A1*B1 + A1*B2 + A2*B1 + A2*B2
  This doesn't help directly.
  
  Another approach: Use the fact that 3 = 2+1 and 4 = 2+2, 6 = 3+3 or other splits.
  
  Actually, let me just implement the schoolbook decomposition correctly. The task asks for
  a rank-minimal exact decomposition, and without a known better algorithm for 3x4x6, the
  schoolbook is the safest choice.
  
  However, let me try one more thing: Can we use the identity for 2x2x2 (Strassen) in a
  block setting?
  
  If we view 3x4x6 as a block matrix multiplication, we might be able to use Strassen's
  algorithm on some of the blocks. But the dimensions don't align nicely for a direct
  application.
  
  Let me just provide the schoolbook decomposition with rank 72. This is guaranteed to be
  correct and exact.
"""

import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Construct the factor matrices for matrix multiplication 3x4x6.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The decomposition satisfies:
    For all i in [0,n), k in [0,p):
        sum_{j=0}^{m-1} A[i,j] * B[j,k] = sum_{r=0}^{rank-1} W[k*n+i, r] * (sum_{j'} U[i*m+j', r] * A[i,j']) * (sum_{j'} V[j'*p+k, r] * B[j',k])
    
    Wait, that's not quite right. Let me re-derive.
    
    The bilinear map is: (A, B) -> C = A * B
    We want: C[i,k] = sum_r (u_r^T A) * (v_r^T B) * w_r[i,k]
    
    Where u_r is a vector in R^{n*m}, v_r is a vector in R^{m*p}, w_r is a vector in R^{n*p}.
    
    In the factor matrix notation:
    U[i*m+j, r] = u_r[i,j]
    V[j*p+k, r] = v_r[j,k]
    W[k*n+i, r] = w_r[i,k]  (note: w_r is indexed by (i,k) but W uses k*n+i)
    
    So: C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
    
    For the schoolbook, each term (i,j,k) contributes:
    u_{(i,j,k)}[i',j'] = delta_{i,i'} delta_{j,j'}
    v_{(i,j,k)}[j',k'] = delta_{j,j'} delta_{k,k'}
    w_{(i,j,k)}[i',k'] = delta_{i,i'} delta_{k,k'}
    
    So: U[i*m+j, r] = 1 if r corresponds to (i,j,k), else 0
        V[j*p+k, r] = 1 if r corresponds to (i,j,k), else 0
        W[k*n+i, r] = 1 if r corresponds to (i,j,k), else 0
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank = 72
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition: each (i,j,k) triple gives one rank-1 term
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i*m + j), representing the (i,j) entry of A
                U[i * m + j][column] = 1
                # V is indexed by (j*p + k), representing the (j,k) entry of B
                V[j * p + k][column] = 1
                # W is indexed by (k*n + i), representing the (i,k) position in C
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
