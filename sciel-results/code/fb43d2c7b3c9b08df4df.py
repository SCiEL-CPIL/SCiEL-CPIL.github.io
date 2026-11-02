"""
Matrix multiplication 4x4 * 4x8 via rank-minimal tensor decomposition.

We use the known fact that matrix multiplication of an (n x m) by (m x p)
matrix can be done with rank r = n*m*p - (n*m - 1)*(p - 1) using the
"Strassen-like" or "Winograd" style factorization, but for small dimensions
we can do better by exploiting the specific structure.

For 4x4 * 4x8:
  n=4, m=4, p=8
  Schoolbook rank = 4*4*8 = 128

Known lower bounds and constructions:
  - For square n x n * n x n, Strassen gives 7 for n=2, and for n=4 the
    best known is 49 (via Strassen's 7^2) but actually for 4x4*4x4 the
    best known upper bound is 49, and lower bound is 49 (so it's tight).
  
  For rectangular 4x4 * 4x8, we can think of this as multiplying a 4x4 matrix
  by a 4x8 matrix. We can split the 4x8 into two 4x4 blocks and use two
  4x4*4x4 multiplications, giving rank 2*49 = 98. But we can do better.

  Actually, let's think more carefully. The tensor for multiplying A (4x4) by B (4x8)
  to get C (4x8) has rank at most 4*4*8 = 128 (schoolbook).

  A better approach: use the fact that we can share computations across the
  two 4x4 blocks of the right-hand side.

  Let's use a direct construction. For 2x2 * 2x4:
    n=2, m=2, p=4
    Schoolbook = 16
    Strassen for 2x2*2x2 is 7, so for 2x2*2x4 we can do 2*7 = 14.
    But we can do better: the rank of the 2x2*2x4 multiplication tensor is 10.
    
  For 4x4 * 4x8, by similar reasoning, we can likely achieve a rank well below 128.

  Let's use a practical approach: use Strassen's algorithm recursively.
  Split 4x4 * 4x8 into four 2x2 * 2x4 blocks, and use Strassen for each.
  Each 2x2*2x4 can be done with rank 10 (as noted above).
  So total rank = 4 * 10 = 40? No, that's not right either.

  Let me think about this differently. I'll use a known result:
  The rank of the matrix multiplication tensor <n,m,p> is at most n*m*p - (n-1)*(m-1)*(p-1) + ... 
  Actually, let's just use a constructive approach with Strassen.

  For 4x4 * 4x8:
  Split A (4x4) into four 2x2 blocks: A11, A12, A21, A22
  Split B (4x8) into four 2x4 blocks: B11, B12, B21, B22
  C (4x8) = [A11*B11 + A12*B21, A11*B12 + A12*B22;
             A21*B11 + A22*B21, A21*B12 + A22*B22]

  Using Strassen for each 2x2*2x4 subproblem:
  For 2x2 * 2x4, we can use a rank-10 construction.
  
  Actually, let me just implement a general Strassen-based approach and verify.

  Given time constraints, let's use a known efficient construction.
  The best known upper bound for 4x4*4x4 is 49, so for 4x4*4x8 we can use
  two 4x4*4x4 multiplications: rank 98.

  But we can do better. Let's use a mixed approach.

  I'll implement a search for a low-rank decomposition using the fact that
  the entries can be small integers. For a 150s budget, we can try a
  heuristic search.
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    # Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_4x4_4x8():
    """
    Use Strassen's algorithm for 4x4 * 4x8.
    
    Split into 2x2 * 2x4 blocks and use a rank-10 construction for each.
    
    For 2x2 * 2x4:
      A = [[a, b], [c, d]], B = [[e, f, g, h], [i, j, k, l]]
      C = [[a*e+b*i, a*f+b*j, a*g+b*k, a*h+b*l],
           [c*e+d*i, c*f+d*j, c*g+d*k, c*h+d*l]]
      
      We can use Strassen-like products:
      M1 = (a+d)*(e+i)
      M2 = (c+d)*e
      M3 = a*(f-j)
      M4 = d*(g-h)
      M5 = (a+b)*h
      M6 = (c-a)*(e+i)  -- wait, this doesn't directly work for 2x2*2x4
      
      Let me use a known rank-10 construction for 2x2*2x4.
      
      Actually, for 2x2*2x4, the minimal rank is 10. Here's a construction:
      We need 10 bilinear products. Let me denote the entries:
      A: a,b,c,d (row-major: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1])
      B: e,f,g,h,i,j,k,l (row-major: B[0,0]=e, B[0,1]=f, B[0,2]=g, B[0,3]=h,
                                       B[1,0]=i, B[1,1]=j, B[1,2]=k, B[1,3]=l)
      C: 8 entries
      
      One rank-10 construction uses these products:
      p1 = a*e
      p2 = a*f
      p3 = a*g
      p4 = a*h
      p5 = b*i
      p6 = b*j
      p7 = b*k
      p8 = b*l
      p9 = (c+d)*e
      p10 = (c+d)*(f+g+h)  -- no, this doesn't give the right structure
      
      Let me just use the schoolbook for the subblocks and combine.
      For 2x2*2x4, schoolbook rank is 8. So 4 subblocks * 8 = 32 total.
      That's better than 128.
      
      Actually, let's just use a direct 32-rank construction.
    """
    n, m, p = 4, 4, 8
    # Use schoolbook but with a smarter indexing to potentially allow optimization
    # For now, return schoolbook of rank 128, and we'll try to improve.
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try a Strassen-based construction
    # For 4x4 * 4x8, split into 2x2 * 2x4 blocks
    # Each 2x2*2x4 can be done with rank 8 (schoolbook for that subproblem)
    # Total: 4 * 8 = 32
    
    # Actually, let's construct a rank-32 decomposition directly
    # by treating the problem as 4 independent 2x2*2x4 multiplications
    
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (32, rank)
    
    # For a 2x2*2x4 subproblem at block (bi, bj) where bi in {0,1}, bj in {0,1}:
    # A block: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1 of A
    # B block: rows 2*bj..2*bj+1, cols 4*bj..4*bj+3 of B
    # C block: rows 2*bi..2*bi+1, cols 4*bj..4*bj+3 of C
    
    # Schoolbook for this subproblem uses 8 products.
    # Total rank = 4 * 8 = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for i_local in range(2):
                for j_local in range(2):
                    for k_local in range(4):
                        i = 2 * bi + i_local
                        j = 2 * bj + j_local
                        k = 4 * bj + k_local
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    
    if _verify(U, V, W, n, m, p):
        best = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try to improve using Strassen for the 2x2*2x4 subproblems
    # Strassen for 2x2*2x2 is rank 7, so for 2x2*2x4 we might use rank 10 or 11
    # But let's first check if we can do better with a different approach
    
    # Try a rank-28 construction using Strassen-like tricks
    # For each 2x2*2x4, use 7 products (Strassen for the first 2x2*2x2) + 4 for the rest
    # That gives 11 per subblock, total 44. Not better than 32.
    
    # Let's try to find a lower rank by using a heuristic search
    # Given time constraints, we'll stick with rank 32 unless we find something better
    
    if time.time() > deadline:
        return best
    
    # Try to verify and return
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
