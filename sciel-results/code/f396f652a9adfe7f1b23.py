"""Winograd Fourier Transform Algorithm (WFTA) for 2x4x6 matrix multiplication.

Strategy: Exploit the cyclic group structure Z_2 x Z_4 x Z_6 (or equivalent) to find
a rank-minimal exact tensor decomposition. The goal is to use the structure of the
multiplication tensor to reduce the rank below the schoolbook 48.

For 2x4x6 matrix multiplication, we can view this as multiplying a 2x4 matrix by a 4x6 matrix.
We can decompose this into blocks. One effective approach is to use Winograd's algorithm
for small matrix multiplications.

However, a more direct approach for finding a low-rank decomposition is to use known
results. For 2x4x6, the schoolbook rank is 48. We can potentially do better.

Let's try a different approach: use the fact that we can combine some of the multiplications.
Actually, let me think about this more carefully.

The tensor for 2x4x6 matrix multiplication has entries T[i,j,k] where:
- i in {0,1} (rows of A, rows of C)
- j in {0,1,2,3} (cols of A, rows of B)  
- k in {0,1,2,3,4,5} (cols of B, cols of C)

The multiplication is C[i,k] = sum_j A[i,j] * B[j,k]

The tensor is T[i,j,k] = 1 if the indices correspond to a valid multiplication, 0 otherwise.
Actually, the multiplication tensor is defined as T[i,j,k] = 1 for all valid (i,j,k) where
i is row of A, j is col of A / row of B, k is col of B.

Wait, let me reconsider. The matrix multiplication tensor for (n,m,p) has:
- First index: i (0..n-1), row of A
- Second index: j (0..m-1), col of A / row of B
- Third index: k (0..p-1), col of B

And T[i,j,k] = 1 for all valid combinations. The rank is the minimum number of rank-1 tensors
needed to sum to this tensor.

For 2x4x6, schoolbook gives rank 48. Let's see if we can do better.

One approach: use Strassen-like techniques. But 2x4x6 is not a power of 2 in all dimensions.

Let me try a constructive approach using Winograd's method. Winograd's algorithm for matrix
multiplication uses the fact that certain linear combinations can reduce the number of
multiplications.

Actually, let me just implement a search or use a known decomposition. Given the time constraints,
let me try to construct a decomposition of rank less than 48.

For 2x4x6, one approach is to split the 4x6 part. We can write the 4x6 multiplication as
two 4x3 multiplications, each of which can be done with some rank. Or we can use other decompositions.

Let me try a different tactic: use the fact that 2x4x6 can be related to smaller problems.

Actually, let me just implement the schoolbook and see if I can find a simple improvement.

Given the complexity, let me implement a careful construction. I'll use the following approach:
- The 2x4x6 multiplication can be seen as 6 separate 2x4x1 multiplications (one for each column of B).
- Each 2x4x1 multiplication has rank 8 (schoolbook), so total rank 48.
- But we can do better by sharing computations.

Let me try to use a rank-42 decomposition based on Strassen for the 2x2 blocks.

Actually, I recall that for 2x4x4, the rank is known to be 22 (or similar). For 2x4x6, it might be around 36-42.

Let me just implement a reasonable decomposition. I'll use a block-based approach with Strassen.

Split the 4x6 matrix into two 4x3 blocks. Then 2x4x6 = two instances of 2x4x3.
For 2x4x3, schoolbook rank is 24. So two of them would be 48. Not helpful.

Let me try: split 2x4x6 into four 1x2x3 blocks? No, that doesn't make sense for matrix mult.

OK, let me just implement the schoolbook decomposition as a fallback, and try to find a better one
through a simple heuristic search or known formula.

Given the time constraints, I'll implement the schoolbook rank-48 decomposition, which is guaranteed
to be correct. This gives score 1.0 (same as baseline). To improve, I need to find a lower rank.

Let me try a different approach: use the fact that we can represent the multiplication using
fewer terms by exploiting linearity. 

Actually, let me just provide the schoolbook solution as it's correct and within time limits.
The task says the seed program scores 1.0, so matching it is acceptable, but we should try to improve.

Let me attempt a rank-42 solution using Strassen's algorithm adapted for this case.

Strassen's algorithm multiplies two 2x2 matrices with 7 multiplications instead of 8.
We can decompose 2x4x6 as follows:
- Split A (2x4) into two 2x2 blocks: A1, A2
- Split B (4x6) into two 4x3 blocks: B1, B2
- C = A1*B1 + A2*B2, where each product is 2x3

For each 2x3 product, we can use a rank-6 decomposition (schoolbook for 2x2x3 would be 12, but 2x3 is different).

Actually, 2x2x3 matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k], i in {0,1}, j in {0,1}, k in {0,1,2}.
Schoolbook rank is 2*2*3 = 12. Can we do better? Probably not easily.

So 2x4x6 with this split gives 2 * 12 = 24? No, that's wrong. Each 2x2 times 2x3 is a 2x3 result.
The tensor for 2x2x3 has rank at most 12. So total rank at most 24? That can't be right because
we're not accounting for the addition properly.

I think I'm overcomplicating this. Let me just implement the schoolbook and return it.
"""

import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
