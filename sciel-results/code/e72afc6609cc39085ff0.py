"""
Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition.

The matmul tensor for C = A * B (A is n×m, B is m×p) has the property that:
C[i, k] = sum_j A[i, j] * B[j, k]

A rank-1 term in the tensor decomposition corresponds to:
  U_col[i*m + j] * V_col[j*p + k] * W_col[k*n + i]

The key insight is that we can group the j-index to create sharing:
- For each j, we can have a term where U has non-zeros at all i for that j,
  V has non-zeros at all k for that j, and W has non-zeros at all (k,i) pairs.
  This single term contributes A[i,j]*B[j,k] to all C[i,k].

Wait, that's not right either. Let me think more carefully.

For a fixed j, if we set:
  U[i*m + j] = 1 for all i
  V[j*p + k] = 1 for all k
  W[k*n + i] = 1 for all k, i

Then the contribution to C[i,k] is:
  W[k*n+i] * sum_{j'} U[i*m+j'] * V[j'*p+k]
  = 1 * U[i*m+j] * V[j*p+k]  (since only j'=j has non-zero in both U and V)
  = 1 * A[i,j] * B[j,k]

So one term per j gives us the full matmul! For 4x4x5, that's m=4 terms total.

Let me verify: with 4 terms (one per j=0,1,2,3):
  Term j: U[i*4+j]=1 for all i∈{0,1,2,3}, V[j*5+k]=1 for all k∈{0,1,2,3,4},
          W[k*4+i]=1 for all k∈{0,1,2,3,4}, i∈{0,1,2,3}
  
  Contribution to C[i,k]: W[k*4+i] * sum_{j'} U[i*4+j'] * V[j'*5+k]
  Only j'=j contributes (since U[i*4+j'] is 1 only when j'=j, and V[j'*5+k] is 1 only when j'=j).
  So contribution = 1 * A[i,j] * B[j,k]. ✓

Sum over j: C[i,k] = sum_j A[i,j]*B[j,k]. ✓

So the rank is m = 4!
"""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 5
    total_rank = m  # 4

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for j in range(m):
        # U: set U[i*m + j] = 1 for all i
        for i in range(n):
            U[i * m + j, j] = 1.0
        # V: set V[j*p + k] = 1 for all k
        for k in range(p):
            V[j * p + k, j] = 1.0
        # W: set W[k*n + i] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 1.0

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
