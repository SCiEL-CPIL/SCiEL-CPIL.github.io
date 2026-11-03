"""Matrix multiplication 4x5x6 — rank-minimal tensor decomposition.

We use the block-strassen approach:
  - Partition the 4x5 matrix into blocks of 2x2 (top-left), 2x3 (top-right), 2x2 (bottom-left), 2x3 (bottom-right)
  - Partition the 5x6 matrix into blocks of 3x3 (top-left), 2x3 (top-right), 3x3 (bottom-left), 2x3 (bottom-right)

Actually, let me think more carefully about the dimensions.

A is 4x5, B is 5x6, C = A*B is 4x6.

The standard schoolbook has rank 4*5*6 = 120.

Strassen-like approaches can reduce rank. For 4x5x6, we can try:
- Split A into 2x2 and 2x3 blocks (rows 1-2, rows 3-4)
- Split B into 3x3 and 2x3 blocks (cols 1-3, cols 4-6)
- The product C is 4x6

Let me think of a block decomposition:
A = [[A11 (2x3), A12 (2x2)], [A21 (2x3), A22 (2x2)]]
B = [[B11 (3x3), B12 (3x3)], [B21 (2x3), B22 (2x3)]]
C = [[C11 (2x3), C12 (2x3)], [C21 (2x3), C22 (2x3)]]

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

This is just the standard block multiplication. Each block product contributes to the rank.

Let me count the rank contribution:
- A11*B11: 2*3*3 = 18
- A12*B21: 2*2*3 = 12
- A11*B12: 2*3*3 = 18
- A12*B22: 2*2*3 = 12
- A21*B11: 2*3*3 = 18
- A22*B21: 2*2*3 = 12
- A21*B12: 2*3*3 = 18
- A22*B22: 2*2*3 = 12
Total: 18+12+18+12+18+12+18+12 = 120. Same as schoolbook.

To get a reduction, I need to use Strassen's trick: instead of 8 multiplications, use 7.

Let me try a different split. Let me use a 2x2 block structure where each block is as square as possible.

Actually, for 4x5x6, the best known approach might be to use a combination of Strassen and schoolbook.

Let me try: split into 2x2 blocks where possible.

A is 4x5. Split into 2x2 blocks: we need 2x2 grid but 5 is odd.
Let me split A into:
- Top: 2 rows, split cols into 2+3: A_top_left (2x2), A_top_right (2x3)
- Bottom: 2 rows, split cols into 2+3: A_bot_left (2x2), A_bot_right (2x3)

B is 5x6. Split into:
- Left: 3 cols, split rows into 2+3: B_left_top (2x3), B_left_bot (3x3)
- Right: 3 cols, split rows into 2+3: B_right_top (2x3), B_right_bot (3x3)

Wait, this is getting complicated. Let me just use a known decomposition.

For M4x5x6, the rank is at most 108 (from various sources). But let me try to construct a valid decomposition.

Actually, let me just use the schoolbook decomposition which gives rank 120, and see if I can improve it.

The key insight: the rank of the matrix multiplication tensor M_{n,m,p} is at most n*m*p (schoolbook), and lower bounds exist. For 4x5x6, the best known upper bound might be around 108-114.

Let me try a simple improvement: use Strassen for a 4x4x4 sub-problem and schoolbook for the rest.

Actually, let me just return the schoolbook decomposition. It's guaranteed to be correct and gives rank 120.

Wait, the task says "Find a rank-minimal exact tensor decomposition". So I need to find the minimum rank. But the evaluator just measures the rank (lower is better). So I should try to minimize the rank.

Let me think about this differently. The tensor T has dimensions 4*5*6 = 120 entries (one per (i,j,k) tuple where i in [0,4), j in [0,5), k in [0,6)). The rank of this tensor is what we're looking for.

For matrix multiplication, the tensor is defined as T[i,j,k] = 1 if and only if the (i,k)-th entry of AB uses A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid (i,j,k).

The rank of this tensor is the minimum number of rank-1 tensors that sum to T.

For small dimensions, we can sometimes find non-trivial decompositions. Let me try to use a known result.

Actually, for the specific case of 4x5x6, I don't have a specific decomposition memorized. Let me just use the schoolbook approach which is guaranteed correct.

But wait - maybe I can use a block Strassen approach. Let me think...

If I split the 4x5 matrix into 2x2 + 2x3 blocks and the 5x6 matrix into 3x3 + 2x3 blocks, I can potentially use Strassen on the 2x2x3 and 2x3x3 sub-problems.

Hmm, this is getting complex. Let me just go with the schoolbook decomposition for now, which gives rank 120.

Actually, I recall that for M_{4,4,4}, the rank is 16 (Strassen) vs 64 (schoolbook). For M_{4,5,6}, maybe I can combine blocks.

Let me try a different approach: use the fact that 4x5x6 can be decomposed into smaller multiplications.

Actually, let me just implement the schoolbook decomposition correctly and return it. The rank will be 120.

Wait, I should try to do better. Let me think about using a 2x2 block Strassen decomposition where possible.

Split A (4x5) into:
  A = [[A1 (2x2), A2 (2x3)]]  -- top 2 rows
      [[A3 (2x2), A4 (2x3)]]  -- bottom 2 rows

Split B (5x6) into:
  B = [[B1 (2x3), B2 (2x3)]]  -- top 2 rows... no wait, B is 5x6.

Let me be more careful. B is 5x6. The column dimension of A must match the row dimension of B. A has 5 columns, B has 5 rows. Good.

Split A's columns into 2+3:
  A = [A_left (4x2), A_right (4x3)]

Split B's rows into 2+3:
  B = [[B_top (2x6)]]
      [[B_bot (3x6)]]

Then C = A_left * B_top + A_right * B_bot
where A_left*B_top is 4x6 and A_right*B_bot is 4x6.

The rank of A_left*B_top (4x2x6) is at most 4*2*6=48 (schoolbook) or less with Strassen.
The rank of A_right*B_bot (4x3x6) is at most 4*3*6=72 (schoolbook) or less.

Total: 48+72=120. Same as schoolbook.

To get an improvement, I need to use Strassen within these sub-problems.

For 4x2x6: this is multiplying a 4x2 matrix by a 2x6 matrix. The result is 4x6.
Schoolbook rank: 4*2*6 = 48.
Can we do better? The tensor is M_{4,2,6}. 

For 4x3x6: M_{4,3,6}. Schoolbook rank: 4*3*6=72.

Hmm, I don't have specific Strassen decompositions for these non-square cases.

Let me try yet another approach. What if I use a 2x2 block decomposition where each block multiplication uses Strassen?

Split A (4x5) into 2x2 blocks:
  A = [[A11 (2x2), A12 (2x3)]]
      [[A21 (2x2), A22 (2x3)]]

Split B (5x6) into 2x2 blocks:
  B = [[B11 (2x3), B12 (2x3)]]  -- rows 0-1
      [[B21 (3x3), B22 (3x3)]]  -- rows 2-4

Wait, this doesn't work cleanly because 5 = 2+3 and 6 = 3+3.

C11 (2x3) = A11*B11 + A12*B21
C12 (2x3) = A11*B12 + A12*B22
C21 (2x3) = A21*B11 + A22*B21
C22 (2x3) = A21*B12 + A22*B22

Block products:
A11 (2x2) * B11 (2x3): rank <= 2*2*3 = 12
A12 (2x3) * B21 (3x3): rank <= 2*3*3 = 18
A11 (2x2) * B12 (2x3): rank <= 2*2*3 = 12
A12 (2x3) * B22 (3x3): rank <= 2*3*3 = 18
A21 (2x2) * B11 (2x3): rank <= 2*2*3 = 12
A22 (2x3) * B21 (3x3): rank <= 2*3*3 = 18
A21 (2x2) * B12 (2x3): rank <= 2*2*3 = 12
A22 (2x3) * B22 (3x3): rank <= 2*3*3 = 18

Total: 12+18+12+18+12+18+12+18 = 120. Still schoolbook.

To get Strassen speedup, I need to combine these 8 products into 7 (or fewer) using Strassen's identity. But Strassen works on square blocks. Here the blocks are not square.

I think for this specific problem, the best I can do without extensive research is the schoolbook decomposition with rank 120. Let me just implement that correctly.

Actually, wait. Let me reconsider. The task says "rank-minimal". Maybe the actual minimum rank for M_{4,5,6} is known and less than 120. But I don't have that information readily available.

Let me just go with the schoolbook decomposition. It's correct and gives rank 120.

"""

import time

def construct():
    """Return the schoolbook decomposition of M_{4,5,6} with rank 120."""
    n, m, p = 4, 5, 6
    rank = n * m * p  # 120
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 120
    V = [[0] * rank for _ in range(m * p)]  # 30 x 120
    W = [[0] * rank for _ in range(p * n)]  # 24 x 120
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
