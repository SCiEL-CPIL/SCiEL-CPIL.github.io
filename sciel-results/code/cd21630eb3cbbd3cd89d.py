"""Matrix multiplication 4,4,6 — rank-minimal exact tensor decomposition.

Strategy:
  • Use the known rank-28 decomposition for 4×4×4 (via Lader/Strassen).
  • Use schoolbook for the 4×4×2 part (rank 32).
  • Total rank = 60, down from schoolbook 96.

The 4×4×4 decomposition uses the fact that rank(4,4,4) ≤ 28.
We construct it by treating the 4×4 matrices as 2×2 blocks of 2×2 matrices,
and using Strassen's algorithm at the block level combined with a rank-7
decomposition for each 2×2×2 block product.

Actually, let me use a simpler correct approach:
  • 4×4×4: rank 64 (schoolbook) — but we can do better
  • Let me just implement schoolbook for 4×4×4 (rank 64) and 4×4×2 (rank 32) = 96 total.

Wait, let me think more carefully. The known bounds:
  rank(2,2,2) = 7 (Strassen)
  rank(4,4,4) ≤ 28 (Lader 1983)
  
For 4×4×6:
  Split into 4×4×4 + 4×4×2
  rank(4,4,4) ≤ 28
  rank(4,4,2) ≤ 32 (schoolbook) or less
  
Actually rank(4,4,2): C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1
  This is two independent 4×4×1 problems... no, they share the same A and B columns.
  rank(4,4,2) = 4*4*2 = 32 for schoolbook.
  Can we do better? The tensor is T[i,j,k] = delta_{j,k} for k in {0,1}, j in {0..3}, i in {0..3}.
  Actually for fixed k, it's the standard 4×4×1 which has rank 16 (each A[i,j]*B[j,k] is one term).
  For two k values: rank 32.
  
  Actually, we can potentially share structure. But let's just use 32.

So total: 28 + 32 = 60.

But I need to actually construct the rank-28 decomposition for 4×4×4.

The Lader construction for rank(4,4,4) ≤ 28:
This is based on the fact that the 4×4×4 multiplication tensor can be embedded in a
structured way. One known construction uses the 2×2×2 Strassen decomposition applied
in a specific way.

Let me use a different approach. I'll construct the rank-28 decomposition explicitly.

The 4×4×4 tensor T has T[i,j,k] = 1 if j==k else 0, for i,j,k in {0,1,2,3}.

A rank-r decomposition means: T[i,j,k] = sum_{s=0}^{r-1} U[i,j,s] * V[j,k,s] * W[k,i,s]

For the rank-28 construction, I'll use the following approach:
- Partition indices into pairs: (0,1) and (2,3)
- Use Strassen's 7-term formula for 2×2×2 at the "block" level
- Each "block" is 2×2, so each block product needs rank 4 (not 8, because we can use
  the 2×2×2 rank-7... wait no)

Actually, let me just use the straightforward construction:

For 4×4×4, think of it as:
  C = A * B where A, B, C are 4×4
  Split into 2×2 blocks: A = [[A00, A01], [A10, A11]], etc.
  C00 = A00*B00 + A01*B10
  C01 = A00*B01 + A01*B11
  C10 = A10*B00 + A11*B10
  C11 = A10*B01 + A11*B11

Each block product like A00*B00 is a 2×2×2 tensor. Using Strassen (rank 7), each block
product takes 7 rank-1 terms. There are 8 block products, so 8*7 = 56 terms. That's not 28.

The rank-28 result comes from a more clever construction that shares terms between
block products. This is non-trivial to construct from scratch.

Let me just use schoolbook for now: rank 96 total. It's correct and safe.

Actually, wait. Let me try a hybrid:
- For the 4×4×4 part: use Strassen at the block level (2×2 blocks of 2×2 matrices)
  This gives 7 block-multiplications, each of which is a 2×2×2 product.
  If I use schoolbook for each 2×2×2 (rank 8): 7*8 = 56
  If I use Strassen for each 2×2×2 (rank 7): 7*7 = 49

- For the 4×4×2 part: schoolbook rank 32

Total: 56 + 32 = 88 (with schoolbook inner) or 49 + 32 = 81 (with Strassen inner)

Both are better than 96! Let me implement the 88-rank version (simpler).

Actually, let me implement the Strassen-at-block-level approach carefully.

4×4×4 via Strassen on 2×2 blocks:
  A = [[A00, A01], [A10, A11]]  (each 2×2)
  B = [[B00, B01], [B10, B11]]  (each 2×2)
  
  Strassen multipliers (scalar level, but each "scalar" is a 2×2 matrix):
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

Each Mi is a product of two 2×2 matrices. The product of two 2×2 matrices is a 2×2×2
tensor multiplication, which has rank 8 (schoolbook) or 7 (Strassen).

For each Mi = X*Y where X and Y are 2×2:
  X = sum of some A-blocks with coefficients
  Y = sum of some B-blocks with coefficients
  
  X*Y = (sum_a ca_a * A_block_a) * (sum_b cb_b * B_block_b)
      = sum_{a,b} ca_a * cb_b * (A_block_a * B_block_b)
  
  Each A_block_a * B_block_b is a 2×2 matrix product → 8 rank-1 terms (schoolbook).

But wait, this expansion leads to many terms. The key insight for Strassen is that
we don't expand — we just compute M1, M2, ..., M7 as intermediate results, and each
intermediate result is a 2×2 matrix. The rank of the whole computation is the sum of
the ranks needed to compute each Mi.

Each Mi = X*Y where X is a 2×2 matrix (linear combo of A-blocks) and Y is a 2×2 matrix
(linear combo of B-blocks). Computing X*Y as a 2×2 matrix product requires rank 8
(schoolbook) or rank 7 (Strassen).

So total rank for 4×4×4 = 7 * 8 = 56 (schoolbook inner) or 7 * 7 = 49 (Strassen inner).

Let me implement the 56-rank version.

For each Strassen multiplier Mi = X*Y:
  X = sum_a ca_a * A_block_a  (2×2 matrix)
  Y = sum_b cb_b * B_block_b  (2×2 matrix)
  
  The product X*Y is a 2×2 matrix. In the tensor decomposition:
  For each element (ii, kk) of the result 2×2 matrix:
    (X*Y)[ii,kk] = sum_jj X[ii,jj] * Y[jj,kk]
    
  X[ii,jj] = sum_a ca_a * A_block_a[ii,jj]
  Y[jj,kk] = sum_b cb_b * B_block_b[jj,kk]
  
  So (X*Y)[ii,kk] = sum_{a,b,jj} ca_a * cb_b * A_block_a[ii,jj] * B_block_b[jj,kk]
  
  Each term A_block_a[ii,jj] * B_block_b[jj,kk] is a rank-1 term in the tensor.

  A_block_a is A[2*ra:2*ra+2, 2*ca:2*ca+2] where (ra,ca) is the block index.
  B_block_b is B[2*rb:2*rb+2, 2*cb:2*cb+2] where (rb,cb) is the block index.
  
  For the term to be valid: the column index of A_block must equal the row index of B_block.
  A_block_a has column block ca, B_block_b has row block rb. We need ca == rb.

  So for a given Mi, the valid terms are those where ca_a == rb_b.

Let me enumerate the Strassen multipliers with their block indices:

M1 = (A00+A11)*(B00+B11)
  A terms: (1, (0,0)), (1, (1,1))  → block indices (row, col) = (0,0) and (1,1)
  B terms: (1, (0,0)), (1, (1,1))  → block indices (row, col) = (0,0) and (1,1)
  Valid pairs (ca == rb):
    (0,0) from A with (0,0) from B: ca=0, rb=0 ✓
    (0,0) from A with (1,1) from B: ca=0, rb=1 ✗
    (1,1) from A with (0,0) from B: ca=1, rb=0 ✗
    (1,1) from A with (1,1) from B: ca=1, rb=1 ✓
  So M1 = 1*1*(A00*B00) + 1*1*(A11*B11) = A00*B00 + A11*B11

M2 = (A10+A11)*B00
  A terms: (1, (1,0)), (1, (1,1))
  B terms: (1, (0,0))
  Valid pairs:
    (1,0) from A with (0,0) from B: ca=0, rb=0 ✓
    (1,1) from A with (0,0) from B: ca=1, rb=0 ✗
  So M2 = 1*1*(A10*B00) = A10*B00

M3 = A00*(B01-B11)
  A terms: (1, (0,0))
  B terms: (1, (0,1)), (-1, (1,1))
  Valid pairs:
    (0,0) from A with (0,1) from B: ca=0, rb=0 ✓
    (0,0) from A with (1,1) from B: ca=0, rb=1 ✗
  So M3 = 1*1*(A00*B01) = A00*B01

M4 = A11*(B10-B00)
  A terms: (1, (1,1))
  B terms: (1, (1,0)), (-1, (0,0))
  Valid pairs:
    (1,1) from A with (1,0) from B: ca=1, rb=1 ✓
    (1,1) from A with (0,0) from B: ca=1, rb=0 ✗
  So M4 = 1*1*(A11*B10) = A11*B10

M5 = (A00+A01)*B11
  A terms: (1, (0,0)), (1, (0,1))
  B terms: (1, (1,1))
  Valid pairs:
    (0,0) from A with (1,1) from B: ca=0, rb=1 ✗
    (0,1) from A with (1,1) from B: ca=1, rb=1 ✓
  So M5 = 1*1*(A01*B11) = A01*B11

M6 = (A10-A00)*(B00+B01)
  A terms: (1, (1,0)), (-1, (0,0))
  B terms: (1, (0,0)), (1, (0,1))
  Valid pairs:
    (1,0) from A with (0,0) from B: ca=0, rb=0 ✓
    (1,0) from A with (0,1) from B: ca=0, rb=0 ✓
    (0,0) from A with (0,0) from B: ca=0, rb=0 ✓
    (0,0) from A with (0,1) from B: ca=0, rb=0 ✓
  So M6 = 1*1*(A10*B00) + 1*1*(A10*B01) + (-1)*1*(A00*B00) + (-1)*1*(A00*B01)
       = A10*B00 + A10*B01 - A00*B00 - A00*B01

M7 = (A01-A11)*(B10+B11)
  A terms: (1, (0,1)), (-1, (1,1))
  B terms: (1, (1,0)), (1, (1,1))
  Valid pairs:
    (0,1) from A with (1,0) from B: ca=1, rb=1 ✓
    (0,1) from A with (1,1) from B: ca=1, rb=1 ✓
    (1,1) from A with (1,0) from B: ca=1, rb=1 ✓
    (1,1) from A with (1,1) from B: ca=1, rb=1 ✓
  So M7 = 1*1*(A01*B10) + 1*1*(A01*B11) + (-1)*1*(A11*B10) + (-1)*1*(A11*B11)
       = A01*B10 + A01*B11 - A11*B10 - A11*B11

Now, the C-blocks are computed as:
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6

Each Mi is a 2×2 matrix. Let me write out the full expansion:

C00 = M1 + M4 - M5 + M7
    = (A00*B00 + A11*B11) + (A11*B10) - (A01*B11) + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    = A00*B00 + A11*B11 + A11*B10 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    = A00*B00 + A01*B10  ✓

C01 = M3 + M5
    = A00*B01 + A01*B11  ✓

C10 = M2 + M4
    = A10*B00 + A11*B10  ✓

C11 = M1 - M2 + M3 + M6
    = (A00*B00 + A11*B11) - A10*B00 + A00*B01 + (A10*B00 + A10*B01 - A00*B00 - A00*B01)
    = A00*B00 + A11*B11 - A10*B00 + A00*B01 + A10*B00 + A10*B01 - A00*B00 - A00*B01
    = A11*B11 + A10*B01  ✓

Great, so the decomposition is correct.

Now, for the tensor decomposition, each block product A_block(ra,ca) * B_block(rb,cb) where ca==rb
contributes 8 rank-1 terms (schoolbook 2×2×2).

The total number of block products across all Mi:
M1: 2, M2: 1, M3: 1, M4: 1, M5: 1, M6: 4, M7: 4 → total 14

But some block products appear in multiple Mi with different coefficients. Let me collect all unique
block products and their total coefficients for each C-block.

Actually, for the tensor decomposition, I don't need to combine them. I just need to list all
rank-1 terms. Each block product A_block(ra,ca)*B_block(rb,cb) (with ca==rb) contributes 8 rank-1
terms to the overall tensor, with a coefficient that depends on which C-block it contributes to.

Wait, I need to be more careful. The tensor T[i,j,k] = delta_{j,k} for the 4×4×4 part.

The decomposition is: T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]

For a block product A_block(ra,ca) * B_block(ca,cb) with coefficient coeff:
  This contributes to C_block(ra,cb).
  The elements are: for ii in {0,1}, jj in {0,1}, kk in {0,1}:
    C[2*ra+ii, 2*cb+kk] += coeff * A[2*ra+ii, 2*ca+jj] * B[2*ca+jj, 2*cb+kk]
  
  This is 8 rank-1 terms:
    For each (ii, jj, kk):
      U row: i*m + j = (2*ra+ii)*4 + (2*ca+jj)
      V row: j*p + k = (2*ca+jj)*6 + (2*cb+kk)
      W row: k*n + i = (2*cb+kk)*4 + (2*ra+ii)
      U[row_u] += coeff
      V[row_v] += 1
      W[row_w] += 1

But wait, the coefficient applies to the whole block product. In the tensor decomposition,
each rank-1 term has U, V, W values. The coefficient goes into U (or V or W).

Let me just enumerate all the terms. I'll create a list of (coeff, ra, ca, cb) tuples for each
block product, then for each, add 8 rank-1 terms.

Let me list all block products with their coefficients and target C-blocks:

From C00 = A00*B00 + A01*B10:
  (1, 0, 0, 0) → C00
  (1, 0, 1, 0) → C00

From C01 = A00*B01 + A01*B11:
  (1, 0, 0, 1) → C01
  (1, 0, 1, 1) → C01

From C10 = A10*B00 + A11*B10:
  (1, 1, 0, 0) → C10
  (1, 1, 1, 0) → C10

From C11 = A10*B01 + A11*B11:
  (1, 1, 0, 1) → C11
  (1, 1, 1, 1) → C11

Wait, that's just schoolbook! 8 block products × 8 rank-1 terms = 64. That's not using Strassen.

The issue is that when I expand the Strassen multipliers and collect terms, I get the same
block products as schoolbook. The Strassen reduction happens at the "computation" level, not
at the "tensor decomposition" level.

For a tensor decomposition, what matters is the number of rank-1 terms. Strassen reduces the
number of multiplications, but in a tensor decomposition, each multiplication is already
decomposed into rank-1 terms.

So for a tensor decomposition of 4×4×4, the minimum rank is 28 (Lader), but constructing it
is non-trivial.

Let me just use schoolbook: rank 64 for 4×4×4 and rank 32 for 4×4×2, total 96.

This is the same as the seed, so score 1.0. But it's correct and safe.

Actually, let me try to find a better decomposition. What if I use the fact that the 4×4×2
part can be optimized?

For 4×4×2: C[i,0] = sum_j A[i,j]*B[j,0], C[i,1] = sum_j A[i,j]*B[j,1]
This is 32 rank-1 terms for schoolbook.

Can I do better? The tensor is T[i,j,k] = delta_{j,k} for k in {0,1}.
For fixed j, the terms are: A[i,j]*B[j,0] for k=0 and A[i,j]*B[j,1] for k=1.
These are independent, so rank = 4*4*2 = 32.

Unless I can share A[i,j] across k values... but in a rank-1 decomposition, each term is
U[i,j,s]*V[j,k,s]*W[k,i,s]. For a fixed s, it only involves one (j,k) pair (since V[j,k,s]
is non-zero for at most one k... no, V is a matrix, so V[j,k,s] can be non-zero for multiple k).

Actually, in a general rank-r decomposition, V[j,k,s] can be non-zero for multiple k. So
there might be sharing. But for the standard tensor T[i,j,k] = delta_{j,k}, the rank is
n*m*p = 32 for n=4, m=4, p=2. I believe this is tight.

So the best I can do with this approach is 64 + 32 = 96.

Let me just implement it correctly.
"""

import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    """Return a rank-96 exact tensor decomposition for 4×4×6 matrix multiply (schoolbook)."""
    t_start = time.time()
    DEADLINE = 120.0  # seconds

    n, m, p = 4, 4, 6
    R = n * m * p  # 96

    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
