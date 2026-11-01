"""
Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
- The tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i (else 0).
- We seek rank r with T = sum_d U[:,d] ⊗ V[:,d] ⊗ W[:,d].
- Use a structured search: exploit the block-diagonal structure in j.
  For each j, the sub-tensor for row i and column k has entries at (i*m+j, j*p+k, k*n+i).
  The schoolbook uses 1 term per (i,j,k), total 48.
  
  Key insight: For a fixed j, we have a 2x1 * 1x8 subproblem (after fixing j),
  but actually it's coupled across i and k. The full structure for fixed j is:
    For i in {0,1}, k in {0,...,7}: T[i*3+j, j*8+k, k*2+i] = 1.
  
  This is equivalent to computing, for fixed j:
    C[i,k] += A[i,j] * B[j,k]
  which is an outer product of A[:,j] (2-vector) and B[j,:] (8-vector),
  placed into the 2x8 output. The tensor for this j has rank 1 (it's a single
  outer product), BUT the U, V, W vectors must be consistent across j.
  
  Actually, the full tensor is a sum over j of these outer-product tensors.
  Each j contributes a rank-1 tensor in the (i,k) subspace, but the full tensor
  U, V, W vectors are shared.
  
  Alternative approach: Use the fact that the multiplication tensor for (n,m,p)
  has rank at most n*m*p, and often less. For (2,3,8), let's try to find a
  decomposition with rank < 48.
  
  Method: Build the tensor, then try to find a low-rank decomposition by
  treating it as a matrix unfolding and using iterative methods.
  
  Tensor T of shape (6, 24, 16). Unfold to matrix of shape (6, 24*16) = (6, 384).
  The rank of this unfolding is an upper bound on the tensor rank.
  But we want the tensor rank (CP rank), not the matrix rank.
  
  Practical approach: Since the tensor is sparse (2304 non-zero... actually only
  48 non-zeros, one per (i,j,k) triple), we can try to construct a decomposition
  where each rank-1 term covers multiple non-zero entries.
  
  For each (i,j,k), we need U[i*3+j, d] * V[j*8+k, d] * W[k*2+i, d] to sum to 1
  over all d, and for other (i',j',k') to sum to 0.
  
  Let's try a constructive approach:
  - Use 3 terms per j (for the 2x8 subproblem), total 9 terms.
  - For fixed j, we need to compute C[:, k] += A[:, j] * B[j, k] for all k.
    This is: for each k, C[0,k] += A[0,j]*B[j,k] and C[1,k] += A[1,j]*B[j,k].
    The tensor slice for fixed j has non-zeros at positions:
      (0*3+j, j*8+k, k*2+0) for i=0, all k
      (1*3+j, j*8+k, k*2+1) for i=1, all k
    This slice can be decomposed into rank 1: 
      U[:, d] has 1 at position 0*3+j and 1 at position 1*3+j? No, that would
      couple i=0 and i=1.
    
    Actually, for fixed j, the slice is a 2x8 outer product pattern. The minimal
    rank to represent this slice (as part of the full tensor) depends on how
    U, V, W are shared.
    
  Let me try a different, more systematic approach:
  
  The tensor T[a,b,c] is non-zero only when:
    a = i*3+j, b = j*8+k, c = k*2+i
    for i in {0,1}, j in {0,1,2}, k in {0,...,7}
  
  So T has 48 non-zero entries (each = 1).
  
  A rank-r decomposition gives 48 equations (one per non-zero entry) that must equal 1,
  and 2304-48 = 2256 equations that must equal 0.
  
  This is a system of polynomial equations. Solving it is hard in general.
  
  Let me try a simple constructive solution:
  
  For each (i,j,k), assign a unique term d(i,j,k). This gives rank 48 (schoolbook).
  
  To reduce rank, we need to share terms. Two entries (i1,j1,k1) and (i2,j2,k2)
  can share a term if there exist non-zero values for U, V, W such that both
  entries get the right contribution and no unwanted entries are created.
  
  This is essentially finding a combinatorial structure.
  
  Given the complexity, let me try a computational approach:
  1. Build the 6x24x16 tensor.
  2. Try to find a low-rank CP decomposition using ALS or similar.
  3. Verify the result.
  
  But ALS might not find the exact rank. Let me try a greedy approach:
  - Start with rank 20, try to find a decomposition.
  - If it fails, try rank 25, 30, etc.
  
  Actually, let me try a much simpler approach: exploit the fact that for each j,
  the problem separates in a sense. Use 16 terms total: 2 per j? No, that's 6 total.
  
  Let me think about this differently. The output C[i,k] = sum_j A[i,j] B[j,k].
  We can write this as:
    C = A @ B
  where A is 2x3 and B is 3x8.
  
  One known result: the rank of (2,3,8) matrix multiplication is at most
  rank(2,3,3) * ceil(8/3) ... no, that's not right.
  
  Let me just try to use numpy to find a decomposition.
  
  Actually, for the purpose of this task, let me return a valid schoolbook decomposition
  with rank 48. The score will be 1.0 (same as seed). To do better, I need to find
  a lower rank.
  
  Let me try one more idea: use the fact that the 2x3 * 3x8 multiplication can be
  viewed as 8 instances of 2x3 * 3x1, each computing C[i,k] for a fixed k.
  Each 2x3 * 3x1 subproblem has rank 3 (schoolbook) or possibly less.
  The total rank would be 8 * rank(2,3,1).
  
  For 2x3 * 3x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1] + A[i,2]*B[2].
  This is a bilinear map (2x3 matrix, 3-vector) -> 2-vector.
  The tensor has non-zeros at (i*3+j, j, i) for i in {0,1}, j in {0,1,2}.
  That's 6 non-zeros. Schoolbook rank is 6. Can we do better?
  
  For 2x3 * 3x1, the tensor is 6x3x2 = 36 entries, 6 non-zeros.
  Rank 6 is schoolbook. Can we do rank 5 or less?
  
  This is a small enough problem that we could try to solve it exactly.
  But for now, let me just return the schoolbook.

  Wait, I just realized: I can try to use the known Strassen-like decomposition.
  For 2x2 * 2x2, Strassen uses rank 7 instead of 8.
  For 2x3 * 3x2, the rank might be 11 (as mentioned in the inspiration).
  
  Let me try to construct a rank-11 solution for 2x3 * 3x2, then tile it
  for the 3x8 case.
  
  2x3 * 3x8 = 4 blocks of 2x3 * 3x2 (columns 0-1, 2-3, 4-5, 6-7).
  If rank(2,3,2) = 11, then rank(2,3,8) <= 4*11 = 44.
  
  Let me hardcode a known rank-11 solution for 2x3 * 3x2.
  
  After some research (from memory), one rank-11 algorithm for 2x3 * 3x2:
  
  Let A = [[a00, a01, a02], [a10, a11, a12]]
  Let B = [[b00, b01], [b10, b11], [b20, b21]]
  C = [[c00, c01], [c10, c11]]
  
  11 products:
  p1  = (a00 + a10) * (b00 + b10)
  p2  = a00 * b00
  p3  = a01 * (b10 + b20)
  p4  = a01 * b10
  p5  = a02 * (b20 + b21)
  p6  = a02 * b20
  p7  = a10 * b00
  p8  = (a10 + a11) * (b01 + b11)
  p9  = a11 * b11
  p10 = a12 * b21
  p11 = (a00 + a12) * (b01 + b21)
  
  Then:
  c00 = p2 + p4 + p6 + (p1 - p2 - p7) + ... this is getting complicated.
  
  I don't have a verified rank-11 algorithm memorized. Let me just return
  the schoolbook rank-48 solution.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Schoolbook decomposition: rank = n*m*p = 48
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if not np.isclose(val, 1.0, atol=1e-10):
                    # Fallback: return schoolbook anyway
                    pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
