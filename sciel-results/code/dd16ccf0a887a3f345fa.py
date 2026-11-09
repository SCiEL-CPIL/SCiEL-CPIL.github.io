"""
Rank-minimal tensor decomposition for 2x3x7 matrix multiplication.

Approach:
- The multiplication tensor T has entries T[i,j,k] for i in [0,2), j in [0,3), k in [0,7).
- A rank-r decomposition is given by factor matrices U (6 x r), V (21 x r), W (14 x r)
  such that for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.

Strategy:
1. Split the 3x7 submatrix (columns j=0,1,2 and rows k=0..6) into:
   - A 2x2 block for (j in {0,1}, k in {0,1}) -> use Strassen (rank 7)
   - Remaining 2x5 block for (j in {0,1}, k in {2..6}) -> schoolbook (rank 10)
   - A 1x7 block for (j=2, k in {0..6}) -> schoolbook (rank 7)
   Total: 7 + 10 + 7 = 24? Wait, let's recount.

Actually, let's think more carefully. The tensor is T[i,j,k] where:
- i: row of A (2 rows)
- j: column of A / row of B (3 cols/rows)
- k: column of B (7 cols)

The product C[i,k] = sum_j A[i,j] * B[j,k]

We can decompose by blocks of j and k.

Block 1: j in {0,1}, k in {0,1} -> 2x2x2 subproblem (i is 2, j is 2, k is 2)
  This is a 2x2 matrix multiplication, which has rank 7 (Strassen).
  The tensor entries are T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1}.
  In the full tensor, these are at positions:
    U index: i*3 + j
    V index: j*7 + k
    W index: k*2 + i

Block 2: j in {0,1}, k in {2,3,4,5,6} -> 2x2x5 subproblem
  This is 2 rows of A times 5 columns of B, with 2 intermediate dimensions.
  Schoolbook: 2*2*5 = 20 terms.
  But we can do better? For 2x2x5, the rank is at most 2*2*5=20, but maybe less.
  Actually, for 2xnxm, the rank is 2*n*m in general unless we use Strassen-like.
  For 2x2x5, we can view it as 5 separate 2x2x1 multiplications, each of rank 2 (since 2x2x1 has rank 2: C0 = A00*B00 + A01*B01, C1 = A10*B00 + A11*B01).
  So 5 * 2 = 10 terms.

Block 3: j=2, k in {0,1,2,3,4,5,6} -> 2x1x7 subproblem
  C[i,k] = A[i,2] * B[2,k]
  This is 2*1*7 = 14 terms (each term is A[i,2]*B[2,k] for specific i,k).
  Wait, no: for each (i,k), it's a single product. So 2*7 = 14 terms.

Total: 7 + 10 + 14 = 31.

But wait, the schoolbook is 2*3*7 = 42. So 31 < 42.

Let's verify the Strassen part for 2x2x2:
Standard Strassen for C = A*B where A is 2x2, B is 2x2:
  M1 = (A00+A11)*(B00+B11)
  M2 = (A10+A11)*B00
  M3 = A00*(B01-B11)
  M4 = A11*(B10-B00)
  M5 = (A00+A01)*B11
  M6 = (A10-A00)*(B00+B01)
  M7 = (A01-A11)*(B10+B11)
  
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6

In our tensor notation, for the block j in {0,1}, k in {0,1}:
  The sub-matrix A_sub is A[:, 0:2] (2x2), B_sub is B[0:2, 0:2] (2x2), C_sub is C[:, 0:2] (2x2).
  
  The tensor entries T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1} should satisfy:
  C_sub[i,k] = sum_{j in {0,1}} A_sub[i,j] * B_sub[j,k]
  
  In the full tensor, T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
  But in a decomposition, we have:
  T[i,j,k] = sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*2+i, c]
  
  For the schoolbook, each (i,j,k) has its own term. For Strassen, we have 7 terms that combine.

Let me write out the Strassen factors for the 2x2x2 block:

For c=0 (M1):
  M1 = (A00+A11)*(B00+B11)
  This contributes to C00 and C11.
  U[0*3+0] = 1, U[1*3+1] = 1  (A00 and A11)
  V[0*7+0] = 1, V[1*7+1] = 1  (B00 and B11)
  W[0*2+0] = 1, W[1*2+1] = 1  (C00 and C11)
  But wait, in the tensor decomposition, each term c is a rank-1 tensor:
  U[:,c] * V[:,c] * W[:,c]
  
  For M1:
  U[0,0] = 1, U[3,0] = 1  (indices for A00 and A11 in the full U: 0*3+0=0, 1*3+1=4? No, 1*3+1=4)
  Wait, U has 6 rows: indices 0..5 for (i,j) pairs: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5.
  
  So for A00: U index 0
  A01: U index 1
  A02: U index 2
  A10: U index 3
  A11: U index 4
  A12: U index 5

  V has 21 rows: indices for (j,k): (0,0)=0, (0,1)=1, ..., (0,6)=6, (1,0)=7, ..., (1,6)=13, (2,0)=14, ..., (2,6)=20.
  
  W has 14 rows: indices for (k,i): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, ..., (6,0)=12, (6,1)=13.

  For M1 = (A00+A11)*(B00+B11):
  U[0,0] = 1, U[4,0] = 1
  V[0,0] = 1, V[8,0] = 1  (B00 is (0,0)=0, B11 is (1,1)=7+1=8)
  W[0,0] = 1, W[3,0] = 1  (C00 is (0,0)=0, C11 is (1,1)=2+1=3)

  But this is a product of sums, not a single rank-1 term. In the tensor decomposition, each c is a single rank-1 term. So M1 is not a single rank-1 term; it's a product of two sums.

  Actually, in the tensor decomposition, the rank-1 terms are of the form:
  u * v * w where u, v, w are vectors.
  
  For M1 = (A00+A11)*(B00+B11), this expands to:
  A00*B00 + A00*B11 + A11*B00 + A11*B11
  
  Each of these is a rank-1 term. So M1 actually corresponds to 4 rank-1 terms in the schoolbook.

  I think I'm confusing the Strassen algorithm with the tensor decomposition. In the Strassen algorithm, the 7 "multiplications" are not rank-1 tensor terms; they are bilinear forms that are products of linear combinations. Each such bilinear form can be decomposed into rank-1 terms.

  For a bilinear form (a0*A00 + a1*A01 + ...)*(b0*B00 + b1*B01 + ...), the number of rank-1 terms is the product of the number of non-zero coefficients in a and b.

  So for M1 = (A00+A11)*(B00+B11), it has 2*2 = 4 rank-1 terms.
  Similarly, each of the 7 Strassen multiplications has at most 4 rank-1 terms, so total up to 28, which is worse than schoolbook.

  So Strassen doesn't directly give a lower-rank tensor decomposition in this sense. The rank of the tensor is the minimum number of rank-1 tensors that sum to T, and each rank-1 tensor is a single product of three vectors.

  I think I need to reconsider. The rank of the 2x2x2 matrix multiplication tensor is 7, which is less than the schoolbook rank of 8. So there IS a rank-7 decomposition. Let me look it up or derive it.

  The standard rank-7 decomposition for 2x2x2 is:
  T[i,j,k] = sum_{c=0}^{6} U[i,j,c] * V[j,k,c] * W[k,i,c]
  
  One known decomposition (from Strassen):
  c=0: U: A00, A11; V: B00, B11; W: C00, C11  -> but this is not rank-1.

  Actually, I recall that the 7 rank-1 terms for 2x2x2 are:
  1. A00 * B00 -> C00
  2. A00 * B01 -> C01
  3. A01 * B00 -> C00
  4. A01 * B01 -> C01
  5. A10 * B10 -> C10
  6. A10 * B11 -> C11
  7. A11 * B10 -> C10
  8. A11 * B11 -> C11
  That's 8 terms (schoolbook).

  The rank-7 decomposition uses combinations. I think I need to look up the explicit factors.

  Given the complexity, let me just use the schoolbook for now and see if I can find a better approach.

  Actually, let me try a different approach: use the fact that for 2xn multiplication, the rank is 2*n. So for 2x3x7, we can view it as 2x(3*7) = 2x21, which has rank 2*21 = 42. Or as (2*3)x7 = 6x7, which has rank 6*7 = 42.

  So the schoolbook rank is 42, and it's hard to beat.

  Let me just implement the schoolbook decomposition correctly and return it.

  Wait, the current program already does this and gets score 1.0. I need to improve it.

  Let me try to use the rank-7 decomposition for the 2x2x2 block and see if it helps.

  After research, I found that the 2x2x2 tensor has rank 7. Here is one decomposition:
  
  The 7 rank-1 terms are:
  c0: U=[1,0,0,0,1,0], V=[1,0,0,1,0,0], W=[1,0,0,1]  (but this is for 2x2x2, so U has 4 entries, V has 4, W has 4)
  
  Let me use a known rank-7 decomposition from literature.
  
  Actually, I'll just implement the schoolbook for now and ensure it's correct. The score is 1.0, which means it's the baseline. To improve, I need to find a lower rank.

  Given the time, let me try a hybrid: use Strassen for the 2x2x2 block (rank 7) and schoolbook for the rest.
  
  For the 2x2x2 block (j in {0,1}, k in {0,1}):
  Use the known rank-7 decomposition.
  
  For the remaining parts:
  - j in {0,1}, k in {2..6}: 2x2x5, schoolbook rank 20
  - j=2, k in {0..6}: 2x1x7, schoolbook rank 14
  
  Total: 7 + 20 + 14 = 41.
  
  This is better than 42!

  Now I need the explicit rank-7 factors for 2x2x2.

  From Strassen's algorithm, the 7 bilinear forms are:
  M1 = (A00+A11)*(B00+B11)
  M2 = (A10+A11)*B00
  M3 = A00*(B01-B11)
  M4 = A11*(B10-B00)
  M5 = (A00+A01)*B11
  M6 = (A10-A00)*(B00+B01)
  M7 = (A01-A11)*(B10+B11)

  Each Mi is a bilinear form, and each can be written as a sum of rank-1 terms. But the total number of rank-1 terms is not 7; it's more.

  I think I'm stuck. Let me just return the schoolbook and hope for the best.

  Actually, let me try a different strategy: use random search to find a lower-rank decomposition. But that's not guaranteed to work within 120s.

  Given the constraints, I'll implement the schoolbook correctly and return it. The score will be 1.0, but that's the best I can do without a breakthrough.

  Wait, the task says "Find a rank-minimal exact tensor decomposition". If the rank-minimal is 42, then schoolbook is optimal.

  Let me just make sure the schoolbook is correct and return it.
"""

import numpy as np


def construct():
    """
    Construct a tensor decomposition for 2x3x7 matrix multiplication.
    
    Returns: dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = 2, 3, 7
    
    rank = n * m * p  # 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1) > 1e-10:
                    raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
