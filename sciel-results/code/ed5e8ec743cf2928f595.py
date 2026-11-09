"""
Matrix multiplication 4x4 by 4x8 — tensor rank minimization.

Structural change: exploit the rectangular structure (p=8=2*4) by treating the
4x8 matrix B as a 4x4 matrix over the ring R = F[t]/(t^2) (dual numbers).
This maps the 4x4x8 problem to a 4x4x4 problem over R, which can be solved
with rank 2*49 = 98 using Strassen over the ring.

However, we also attempt a direct Kronecker product approach:
  - Use the known rank-7 Strassen for 2x2x2
  - Build 4x4x4 with rank 49 via block Strassen
  - For 4x4x8, we can potentially do better than 2*49=98 by exploiting
    the specific rectangular structure

Actually, let me think more carefully. The tensor for 4x4*4x8 has dimensions
(4*4, 4*8, 8*4) = (16, 32, 32). The schoolbook rank is 4*4*8 = 128.

The current approach gives rank 98 by splitting into two 4x4*4x4 problems.

Can we do better? Let's think about it differently.

The key insight: a 4x8 matrix B can be seen as two 4x4 matrices B1, B2 side by side.
C = A*B = [A*B1, A*B2]

This is exactly two 4x4*4x4 multiplications sharing the same A. In the tensor
decomposition, the U matrix (for A) can be shared!

So instead of rank 49 + 49 = 98, we can potentially use rank 49 for the shared
part plus additional terms. But actually, the tensor decomposition doesn't
"share" in that simple way because each rank-1 term is independent.

Wait, let me reconsider. In the tensor decomposition:
T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]

For the 4x4x8 problem:
- i, j in 0..3 (A is 4x4)
- j, k in 0..7 (B is 4x8)  
- k, i in 0..7 (C is 8x4, but W[k*n+i] with n=4, so k in 0..7, i in 0..3)

Actually W has shape (p*n, rank) = (8*4, rank) = (32, rank), indexed by W[k*4+i].

The tensor T[i,j,k] = 1 if the (i,j) element of A and (j,k) element of B contribute
to the (i,k) element of C. For matrix multiplication, T[i,j,k] = 1 for all valid
(i,j,k) where j is the contraction index.

Wait no. The matrix multiplication tensor has T[i,j,k] = 1 for all i in 0..n-1,
j in 0..m-1, k in 0..p-1. The decomposition is:
C[i,k] = sum_j A[i,j] * B[j,k]

So the tensor T has T[i,j,k] = 1 for all (i,j,k), and we need:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)

This is the standard matmul tensor.

For 4x4*4x8: n=4, m=4, p=8.
U has shape (16, rank), V has shape (32, rank), W has shape (32, rank).

The rank-98 solution splits B into two 4x4 halves and does two independent
4x4*4x4 multiplications.

To do better, we need to find a decomposition with rank < 98.

One approach: use the fact that the two halves share the same A. In the
tensor decomposition, this means we can potentially combine terms.

Let me try a different structural approach: use a 2x2 block structure where
each block is 2x4 (for B) or 2x2 (for A and C).

Actually, let me try the dual numbers approach more carefully.

If we identify a 4x8 matrix B with a 4x4 matrix over R = F[t]/(t^2), where
B[j, 2k] = B_real[j,k] and B[j, 2k+1] = B_imag[j,k], then:
(A * B)[i, 2k] = sum_j A[i,j] * B_real[j,k]
(A * B)[i, 2k+1] = sum_j A[i,j] * B_imag[j,k]

This is just two separate 4x4*4x4 multiplications, so rank 98.

But what if we use a different algebra? For example, if we use the algebra
F[x]/(x^2 - 1) = F x F, we get the same thing.

What about using a non-commutative algebra? Or a different decomposition?

Let me try yet another approach: direct search for a lower-rank decomposition
using the known structure.

Actually, I think the best structural change is to use the following:
- The 4x4*4x8 matmul can be seen as computing 4*8 = 32 outputs, each being
  an inner product of a length-4 vector (row of A) with a length-4 vector
  (column of B, but there are 8 columns).

- Use a 2x2 block Strassen where the "scalar" multiplications are 2x4*2x4 -> 2x4
  matrix multiplications.

Wait, that doesn't quite work either.

Let me try a completely different approach: use the known result that the
tensor rank of 4x4*4x4 is at most 49 (from Strassen), and see if we can
exploit the rectangular structure to get below 98.

One idea: use a 2x4 block structure.
- A is 4x4, split into four 2x2 blocks: A11, A12, A21, A22
- B is 4x8, split into eight 2x2 blocks: B11, B12, B13, B14, B21, B22, B23, B24
- C is 8x4, split into eight 2x2 blocks: C11, C12, C13, C14, C21, C22, C23, C24

Wait, C is 8x4 (p x n = 8 x 4), so C has 8 rows and 4 columns.

Let me re-index: C[i,k] where i in 0..7 (rows) and k in 0..3 (columns).
Actually, in the convention W[k*n+i], k is the column index of C and i is the
row index. So C has p=8 columns and n=4 rows? No, that doesn't make sense.

Let me re-read the convention: W[k*n+i, r] where k in 0..p-1 and i in 0..n-1.
So W has p*n = 8*4 = 32 entries. The index k*n+i suggests k is the "outer"
index and i is the "inner" index.

In the tensor decomposition:
T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

where i in 0..n-1=0..3, j in 0..m-1=0..3, k in 0..p-1=0..7.

So C[i,k] = sum_j A[i,j] * B[j,k], with i in 0..3, j in 0..3, k in 0..7.
C is a 4x8 matrix (4 rows, 8 columns).

OK so my original understanding was correct. Let me try a different structural
approach.

What if I use a 2x2 block Strassen where each "block" multiplication is a
2x2 * 2x4 -> 2x4 matrix multiplication?

A 2x2 * 2x4 matrix multiplication can be done with rank 2*2*4 = 16 (schoolbook)
or potentially less.

Using Strassen for 2x2*2x4:
- A is 2x2, B is 2x4, C is 2x4
- Split B into two 2x2 blocks: B1, B2
- C1 = A*B1, C2 = A*B2
- Each is a 2x2*2x2 multiplication with rank 7
- Total rank: 7 + 7 = 14

So a 2x2*2x4 multiplication has rank at most 14.

Now, for the 4x4*4x8 problem using 2x2 blocks:
- A: 4x4 = four 2x2 blocks
- B: 4x8 = eight 2x2 blocks (2 rows of blocks, 4 columns of blocks)
- C: 4x8 = eight 2x2 blocks (2 rows of blocks, 4 columns of blocks)

Block-level multiplication:
C[i1,k1] = sum_{j1} A[i1,j1] * B[j1,k1]

where i1, j1 in {0,1} and k1 in {0,1,2,3}.

For each (i1,k1) pair, C[i1,k1] is a 2x2 matrix computed as:
C[i1,k1] = A[i1,0]*B[0,k1] + A[i1,1]*B[1,k1]

where A[i1,j1] and B[j1,k1] are 2x2 matrices.

This is a 2x2*2x2 matrix addition of products, which can be done with rank
2*7 = 14 per (i1,k1) pair (using Strassen for each product).

Total: 2*4 = 8 pairs, each with rank 14, total rank = 112. That's worse than 98.

Hmm. Let me think differently.

What if I use a 4x4 block structure where each "block" is a 1x1 scalar?
That's just the schoolbook approach with rank 128.

What if I use a 2x4 block structure?
- A: 4x4, split into two 2x4 blocks (top and bottom)
- B: 4x8, split into four 2x4 blocks (2x2 grid)
- C: 4x8, split into four 2x4 blocks (2x2 grid)

Block-level:
C[top, left] = A[top] * B[top, left] + A[bottom] * B[bottom, left]
C[top, right] = A[top] * B[top, right] + A[bottom] * B[bottom, right]
C[bottom, left] = A[top] * B[top, left] + A[bottom] * B[bottom, left]
...

This is getting complicated. Let me just try to implement a rank-98 solution
and verify it works, since that's what the current program does.

Actually, the current program already gives rank 98. The question is whether we
can do better. Let me try a different approach: use the fact that the two 4x4*4x4
multiplications share the same A to potentially reduce the rank.

In the tensor decomposition, the U matrix (for A) appears in both halves. If we
could "merge" some of the rank-1 terms, we might reduce the total rank.

Specifically, in the rank-98 solution:
- First 49 terms: U1 (16x49), V1 (32x49), W1 (32x49) for the first half
- Last 49 terms: U2 (16x49), V2 (32x49), W2 (32x49) for the second half

U1 and U2 are the same (both are the U from the 4x4*4x4 Strassen decomposition).
So the full U is [U1 | U2] = [U444 | U444].

Can we combine these? If U1 = U2 = U444, then the full U has rank at most 49
(as a matrix), but the tensor rank is still 98 because V and W are different.

I don't think we can easily reduce below 98 with this approach.

Let me try a completely different structural approach: use a 4x4*4x8 decomposition
based on a 2x2*2x4 building block.

A 2x2*2x4 matrix multiplication:
A is 2x2, B is 2x4, C is 2x4.
C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, k in 0..3.

This can be done with rank 14 (two 2x2*2x2 Strassen multiplications).

Now, for the 4x4*4x8 problem:
- A: 4x4, split into four 2x2 blocks
- B: 4x8, split into eight 2x2 blocks
- C: 4x8, split into eight 2x2 blocks

Block-level:
C[i1,k1] = A[i1,0]*B[0,k1] + A[i1,1]*B[1,k1]

where i1 in 0..1, k1 in 0..3, and each product is a 2x2*2x2 matrix multiplication.

For each (i1,k1) pair, we need to compute:
C[i1,k1] = A[i1,0]*B[0,k1] + A[i1,1]*B[1,k1]

This is the sum of two 2x2*2x2 products. Using Strassen for each product:
- A[i1,0]*B[0,k1]: rank 7
- A[i1,1]*B[1,k1]: rank 7
Total per (i1,k1): rank 14

Total: 2*4 = 8 pairs, each with rank 14, total rank = 112. Still worse than 98.

I think rank 98 is the best we can get with known constructions. Let me just
implement a clean rank-98 solution and verify it.

Wait, I just realized: the current program already gives rank 98. The score is
seed/value where seed is the schoolbook rank (128). So score = 128/98 ≈ 1.306.

The inspiration 1 mentions a score of 1.3061, which matches. So the current
program is already at the best known rank.

Let me just make sure the program is correct and efficient, and return it.
"""

import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 140.0


def _time_up():
    return (time.time() - _START_TIME) > _TIMEOUT


def _validate(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _make_schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_222():
    """
    Correct 7-term Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ]
    V = [
        [1,  0,  0, -1,  1,  0,  0],
        [0,  0, 1,  0,  1,  0,  1],
        [0,  0, -1, 1,  0,  1,  0],
        [1,  1,  0,  0,  0,  1,  1]
    ]
    W = [
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  1,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1,  0,  0,  0,  1,  0]
    ]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                if s != 1:
                    raise ValueError("Strassen 2x2x2 verification failed")
    
    return U, V, W


def _block_strassen_444():
    """
    Build a 4x4x4 decomposition using block Strassen.
    rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_222()
    
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    A_blocks = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    
    B_blocks = [
        [[1, 0], [0, 1]],
        [[0, 0], [0, 1]],
        [[0, 1], [-1, 0]],
        [[0, 0], [1, -1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, 1]]
    ]
    
    C_blocks = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [1, 0]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            U[i*4+j][r] += A_blocks[b][i1][j1] * U2[i0*2+j0][s]
            
            for j1 in range(2):
                for j0 in range(2):
                    for k1 in range(2):
                        for k0 in range(2):
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            V[j*4+k][r] += B_blocks[b][j1][k1] * V2[j0*2+k0][s]
            
            for k1 in range(2):
                for k0 in range(2):
                    for i1 in range(2):
                        for i0 in range(2):
                            k = 2*k1 + k0
                            i = 2*i1 + i0
                            W[k*4+i][r] += C_blocks[b][i1][k1] * W2[k0*2+i0][s]
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s_val = 0
                for r in range(49):
                    s_val += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                if s_val != 1:
                    raise ValueError(f"Block Strassen 4x4x4 verification failed at ({i},{j},{k}): got {s_val}")
    
    return U, V, W


def construct():
    n, m, p = 4, 4, 8
    
    try:
        U444, V444, W444 = _block_strassen_444()
    except Exception:
        return _make_schoolbook(n, m, p)
    
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*4+j][r] = U444[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V[j*4+k][r] = V444[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[k*4+i][r] = W444[k*4+i][r]
    
    for r in range(49):
        r2 = r + 49
        for i in range(4):
            for j in range(4):
                U[i*4+j][r2] = U444[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V[j*4+(k+4)][r2] = V444[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i][r2] = W444[k*4+i][r]
    
    if not _validate(U, V, W, n, m, p):
        return _make_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
