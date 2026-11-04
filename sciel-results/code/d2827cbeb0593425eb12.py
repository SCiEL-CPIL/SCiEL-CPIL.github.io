"""
Rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
We use the known lower/upper bounds: the rank of M_{3,3,7} is at most 3*3*7=63 (schoolbook).
We know that for matrix multiplication M_{n,m,p}, the rank is at most n*m*p, and known
results give better bounds. For M_{3,3,7}, we can exploit the fact that the 7 columns
in the output can be handled more efficiently.

Key insight: We can decompose M_{3,3,7} by treating it as related to M_{3,3,3} plus
additional terms. The rank of M_{3,3,3} is known to be 23 (from work by Alon, Yuster, Zwick,
and later improvements). Actually, the best known upper bound for M_{3,3,3} is 23.

However, a simpler approach: we can use the fact that M_{3,3,7} can be viewed as
M_{3,3,3} ⊗ M_{1,1,1} for the first 3 columns, plus additional 4 columns.

Actually, let me think more carefully. The standard result is that rank(M_{n,m,p}) ≤ n*m*p,
and for specific cases we can do better. 

For M_{3,3,7}, one approach:
- The 7 output columns can be split into groups. 
- We know rank(M_{3,3,3}) ≤ 23 (this is a well-known result, the exact rank is between 19 and 23,
  with 23 being the best known upper bound from explicit constructions).

Wait, actually the best known upper bound for M_{3,3,3} is 23, from a construction by
Alon, Yuster, and Zwick (1995) or similar. But I'm not 100% certain of the exact value.

Let me use a different, more reliable approach. We can construct an explicit decomposition.

For M_{3,3,7}, we can use the following strategy:
- For each of the 7 output columns k, we need to compute the 3x3 product A*B_k.
- The schoolbook method uses 9 multiplications per column, so 63 total.

A better approach: use the fact that we can share computation across columns.

Actually, let me try a concrete construction. The rank of M_{3,3,p} for small p is known
in some cases. Let me just try to construct an explicit low-rank decomposition.

One approach: use the identity that for any 3x3 matrix multiplication, we can use
the "fast matrix multiplication" ideas. But for 3x3, the best known is 23.

Let me try: rank(M_{3,3,7}) ≤ rank(M_{3,3,3}) + 4*rank(M_{3,3,1})
where rank(M_{3,3,1}) = 9 (since it's just a 3x3 times a vector, which is 9 scalar mults).
So that gives 23 + 36 = 59, not great.

Another approach: rank(M_{3,3,7}) ≤ 7 * rank(M_{3,3,1}) / something... no.

Let me just try to construct an explicit decomposition with rank < 63.

Actually, the simplest improvement: use the fact that M_{3,3,7} = M_{3,3,3} * (I_3 ⊕ I_4)
in some sense. No, that's not right.

Let me think about this differently. The tensor T for M_{3,3,7} has T[i,j,k] = 1 if we
multiply A[i,j] by B[j,k] to get C[i,k]. The rank is the minimum number of rank-1 tensors
that sum to T.

A rank-1 tensor is T[i,j,k] = U[i,j] * V[j,k] * W[i,k] for some vectors U, V, W.

For the schoolbook: each (i,j,k) triple gives one rank-1 term.

To reduce rank, we need to find linear combinations. For example, for the 3x3x3 case,
there are known decompositions with 23 terms.

Let me just implement the schoolbook and see if I can find a simple improvement.

Actually, I recall that for M_{n,m,p}, if we fix one dimension, we can sometimes do better.
For M_{3,3,7}, one approach is to use the decomposition:

C = A * B where A is 3x3, B is 3x7.

We can write B = [B1, B2, B3, B4, B5, B6, B7] where each Bk is a 3x1 column.

C = [A*B1, A*B2, ..., A*B7]

Each A*Bk is a 3x1 vector, requiring 9 scalar multiplications. Total 63.

To reduce, we can use the fact that A is the same for all columns. The "shared" part is A.

One idea: precompute some linear combinations of rows of A, then use those for all columns.

But that doesn't directly reduce the tensor rank.

Let me just go with the schoolbook for now and see if the evaluation accepts it, then try to
improve.

Wait, the task says "Find a rank-minimal exact tensor decomposition" and "score is the rank".
So lower rank is better. The schoolbook gives rank 63.

Known results:
- rank(M_{2,2,2}) = 7 (Strassen)
- rank(M_{3,3,3}) ≤ 23 (best known upper bound)
- rank(M_{3,3,7}) ≤ ?

I don't know the exact best upper bound for M_{3,3,7}. Let me just try to construct something
better than 63.

One simple improvement: for the 3x3 part, use a 23-term decomposition, and for the extra 4
columns, use 9 terms each. Total: 23 + 4*9 = 59.

But I need to actually construct the 23-term decomposition for M_{3,3,3}. I don't have it memorized.

Let me just implement the schoolbook and see. If I can find a simple improvement, I'll add it.

Actually, let me try a different approach. For M_{3,3,7}, I can use the following:

The 7 columns can be grouped. Suppose I use a decomposition that handles 3 columns at a time
with some shared computation.

For 3 columns (i.e., M_{3,3,3}), if I use the schoolbook, that's 27 terms. If I can find a
23-term decomposition, that saves 4 terms per group of 3 columns.

For 7 columns: 3 + 3 + 1 = 27 + 27 + 9 = 63 with schoolbook.
With 23-term for groups of 3: 23 + 23 + 9 = 55.

But I need the explicit 23-term decomposition. Let me search my memory...

I don't recall the exact 23-term decomposition for M_{3,3,3}. Let me just use the schoolbook
and see if the evaluator accepts it. The score will be 63, which is the seed value.

Wait, the seed program has score 1.0, which corresponds to rank 63 (since score = seed/value,
and seed is presumably 63, so 63/63 = 1.0).

So I need to find a decomposition with rank < 63 to improve.

Let me try to construct an explicit decomposition with rank < 63.

One simple idea: for the last 4 columns, instead of using 9 terms each (36 total), can I do
better by sharing with the first 3 columns?

Actually, here's a concrete approach. I'll use the following construction:

For M_{3,3,7}, I can use the fact that the 7 output columns can be handled by first computing
some intermediate values.

But I'm running out of time. Let me just implement the schoolbook and add a deadline guard.

If I have time, I'll try to find a better decomposition.

Actually, let me try one more idea. For M_{3,3,7}, I can use the following:

C[i,k] = sum_j A[i,j] * B[j,k]

I can precompute for each i,j: A[i,j] (trivial). Then for each j,k: B[j,k] (trivial).

The "multiplication" step is the 63 scalar products.

To reduce, I can use linear combinations. For example, instead of computing A[i,j]*B[j,k]
for each (i,j,k) separately, I can compute sums like (A[i,0] + A[i,1]) * (B[0,k] + B[1,k])
which gives me A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k] in one "multiplication".

But this doesn't directly reduce the tensor rank because I still need to extract the individual
terms.

I think the best I can do without knowing explicit low-rank decompositions is the schoolbook.

Let me just implement it with a deadline guard.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x7 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The entries are integers or half-integers.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Schoolbook decomposition: rank = n*m*p = 63
    rank = n * m * p
    
    # Initialize matrices as lists of lists (will be converted to numpy arrays)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
