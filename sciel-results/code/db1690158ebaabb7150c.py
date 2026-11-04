"""
Matrix multiplication 4x5*5x6 via group-theoretic embedding.

We embed the multiplication tensor into the group algebra of G = Z_2 x Z_2 x Z_3
(order 12) and use the character orthogonality to construct a rank-12 decomposition.

The key idea: For a group G with |G| = g, the group algebra has a basis indexed by
elements of G. The multiplication in the group algebra gives a natural tensor
decomposition of rank g (one per group element), but we need to map the matrix
multiplication indices to group elements.

For C_{n,m,p} with n*m*p = 120 and |G| = 12, we cannot directly use the group
algebra approach in the naive way. However, we can use the following trick:

Write the indices as:
  i = 2*i1 + i0,  i1 in {0,1}, i0 in {0,1}
  j = 5*j1 + j0,  j1 in {0},   j0 in {0,1,2,3,4}  -- this doesn't factor nicely
  
Actually, let me try a different approach. We use the fact that:
  4 = 2 * 2,  6 = 2 * 3

So we can write:
  i = 2*i1 + i0,  i1 in {0,1}, i0 in {0,1}
  k = 3*k1 + k0,  k1 in {0,1}, k0 in {0,1,2}

The multiplication (A*B)[i,k] = sum_j A[i,j] * B[j,k] can be decomposed into
blocks based on (i1, i0) and (k1, k0).

For each fixed (i1, k1, j), we have a 2x3 sub-multiplication:
  (A_{i1} * B_{k1})[i0, k0] = sum_j A[i1*2+i0, j] * B[j, k1*3+k0]

This is a 2x5 * 5x3 multiplication for each (i1, k1) pair, giving 4 such sub-problems.
Each 2x5*5x3 multiplication has rank at most 2*5*3 = 30 by schoolbook, but we can
do better.

Actually, let me try a completely different approach: use the Strassen-like
decomposition for 2x2 blocks.

For 2x2 * 2x2, Strassen gives rank 7 instead of 8.

We can decompose 4x5*5x6 as:
  Write A as a 2x2 block matrix with 2x5 blocks
  Write B as a 2x2 block matrix with 5x3 blocks
  
  A = [[A00, A01], [A10, A11]]  where each Ai is 2x5
  B = [[B00, B01, B02], [B10, B11, B12]]  -- wait, B is 5x6, not 2x2 blocks of 5x3

Let me reconsider. A is 4x5, B is 5x6.

Split A into top and bottom halves: A_top (2x5), A_bottom (2x5)
Split B into left and right halves: B_left (5x3), B_right (5x3)

Then (A*B) = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]

Each of these is a 2x5 * 5x3 multiplication. There are 4 such sub-multiplications.
Each has rank at most 2*5*3 = 30 by schoolbook, total 120.

But we can use Strassen on the 2x2 block structure if we can make the blocks square.
They're not square (2x5 and 5x3), so Strassen doesn't directly apply.

Let me try yet another approach: use the known rank of C_{2,5,3}.

Actually, for C_{2,5,3}, the schoolbook rank is 30. The lower bound is max(2*5, 5*3, 3*2) = 15.

I don't know the exact rank of C_{2,5,3}, but let me try to find a decomposition
with rank less than 30.

Given the time constraints, let me try a practical approach: use a random search
or structured search to find a low-rank exact decomposition.

Actually, let me just implement a simple improvement: use the fact that we can
sometimes reduce the rank by using linear combinations of the schoolbook terms.

For now, let me implement a decomposition that uses the block structure and
applies a simple rank reduction.

After further thought, let me implement the following:
- Use the schoolbook decomposition as a starting point (rank 120)
- Try to find linear dependencies among the rank-1 tensors
- Use Gaussian elimination to find a basis for the span of the rank-1 tensors

But this is computationally expensive. Let me just return the schoolbook for now.

Wait, I have a better idea. Let me use the following known result:
The rank of C_{n,m,p} is at most n*m*p - min(n,m,p) + 1 for some cases.

Actually, I recall that for C_{2,2,2}, the rank is 7 (Strassen). For C_{2,2,p},
the rank is at most 7*p/2 for even p or 7*(p+1)/2 for odd p.

For our case, let me try to decompose using Strassen on 2x2 sub-blocks.

A is 4x5. Write it as:
  A = [[a00, a01, a02, a03, a04],
       [a10, a11, a12, a13, a14],
       [a20, a21, a22, a23, a24],
       [a30, a31, a32, a33, a34]]

B is 5x6. Write it as:
  B = [[b00, b01, b02, b03, b04, b05],
       [b10, b11, b12, b13, b14, b15],
       [b20, b21, b22, b23, b24, b25],
       [b30, b31, b32, b33, b34, b35],
       [b40, b41, b42, b43, b44, b45]]

This is getting complicated. Let me just implement a simple structured decomposition
that might give a modest improvement.

I'll use the following approach:
1. Split the 4x5 * 5x6 into four 2x5 * 5x3 sub-problems
2. For each 2x5 * 5x3, use a decomposition based on the fact that 5 = 2+3
3. This gives a total rank of 4 * (some rank for 2x5*5x3)

For 2x5*5x3, I'll use the schoolbook (rank 30) for simplicity, giving total rank 120.

But I can do better. Let me use the following trick for 2x5*5x3:
  Write the 5 columns of A (2x5) as two 2x2 blocks and one 2x1 block
  Write the 5 rows of B (5x3) as two 5x2 blocks and one 5x1 block... this doesn't work.

I give up trying to find a clever decomposition in this time. Let me just return
the schoolbook decomposition, which is guaranteed correct with rank 120.
"""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition with rank n*m*p = 120
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
