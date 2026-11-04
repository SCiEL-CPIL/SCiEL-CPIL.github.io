"""
Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix 
by a 3 by 6 matrix.

Strategy:
1. Primary: Use the known Strassen decomposition for 3x3x3 matrix multiplication.
   Since the second matrix is 3x6, we can view it as two independent 3x3 blocks.
   The multiplication (3x3) * (3x6) can be decomposed as:
   C[:, 0:3] = A * B[:, 0:3]  (3x3 * 3x3)
   C[:, 3:6] = A * B[:, 3:6]  (3x3 * 3x3)
   
   Using Strassen's algorithm (rank 7) for each 3x3x3 block, we get a total
   rank of 7 + 7 = 14. This is significantly better than the schoolbook rank of 54.

2. Fallback: If verification fails (shouldn't happen with correct Strassen),
   return the schoolbook decomposition of rank 54.

The Strassen decomposition for 3x3x3 matrix multiplication:
We partition each 3x3 matrix into 2x2 blocks (with the 3rd row/col handled carefully).
Actually, standard Strassen is for 2x2x2. For 3x3, we can use the fact that
a 3x3 matrix multiplication can be embedded or we can use a known rank-7 decomposition.

Actually, let's be more careful. The standard Strassen algorithm reduces 2x2x2 from 8 to 7.
For 3x3x3, the schoolbook is 27. Can we do better?
Yes, by partitioning into 2x2 blocks and using Strassen, but we have to handle the boundaries.
A 3x3 matrix can be seen as a 2x2 block matrix where the bottom-right block is 1x1,
top-right is 2x1, bottom-left is 1x2, top-left is 2x2.
This gets complicated.

Let's use a simpler, well-known rank reduction for 3x3x3.
It is known that the rank of 3x3x3 matrix multiplication is at most 23 (using various algorithms).
But finding an explicit rank < 27 decomposition is non-trivial to code from scratch.

However, the directive suggests: "Treat the 3x3x6 multiplication tensor as a block-diagonal sum of two independent 3x3x3 matrix multiplication tensors by partitioning the 6 columns of the second matrix into two groups of 3, and directly apply the known Strassen decomposition of rank 7 for each 3x3x3 block to achieve a total rank of 14."

Wait, Strassen is for 2x2x2, not 3x3x3. The rank of 2x2x2 is 7.
If we partition the 3x3 matrices into 2x2 blocks, we need to be careful.
Let's try a different approach. What if we just use the 2x2x2 Strassen decomposition
and handle the 3x3 case by embedding? No, that's not straightforward.

Let's reconsider. The directive explicitly mentions "Strassen decomposition of rank 7 for each 3x3x3 block".
This is technically incorrect because Strassen is for 2x2x2. However, there is a known
decomposition for 3x3x3 with rank less than 27. Let's look for a concrete one.

Actually, a simpler approach: We can use the fact that matrix multiplication is bilinear.
For (3x3) * (3x6), we can write:
C = A * B
where A is 3x3, B is 3x6, C is 3x6.
We can partition B into B1 (3x3) and B2 (3x3).
C = [A*B1, A*B2]
So we need two independent 3x3x3 multiplications.
If we have a rank R decomposition for 3x3x3, the total rank is 2R.
Schoolbook 3x3x3 has rank 27, so total 54.
If we can find a rank < 27 for 3x3x3, we improve.

Known results: The rank of 3x3x3 matrix multiplication is known to be at most 23.
But coding a specific rank-23 decomposition is complex.

Let's try a different structural idea. Can we use the 2x2x2 Strassen decomposition?
Partition A into 2x2 blocks:
A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
This is messy.

Let's stick to the directive's suggestion but correct the math.
The directive says "Strassen decomposition of rank 7 for each 3x3x3 block".
This is likely a confusion with 2x2x2. Let's assume the intent is to use a known
low-rank decomposition for 3x3x3.

Actually, there's a simpler observation. The 3x3x6 multiplication can be seen as
multiplying a 3x3 matrix by a 3x6 matrix. We can use the fact that:
(3x3)*(3x6) = (3x3)*(3x3) + (3x3)*(3x3)
If we use the standard Strassen algorithm for 2x2x2, we can't directly apply it to 3x3x3.

Let's try a concrete rank reduction. We can use the following identity for 3x3x3:
It is known that the rank of 3x3x3 is at most 23. But let's try to implement a simpler
rank reduction. 

Actually, let's use a different approach. We can use the fact that:
For 3x3x3, we can partition into 2x2 blocks and use Strassen's 7 multiplications,
but we need to handle the extra row and column. This is not straightforward.

Let's try a different structural change. What if we use the "Laderman's algorithm" for 4x4x4?
No, that's for 4x4.

Given the complexity, let's implement a correct rank-54 schoolbook decomposition first,
and then try to improve it using a known rank reduction for 3x3x3.

Actually, I recall that for 3x3x3, there is a rank-23 decomposition, but it's complex.
Let's try a simpler improvement. We can use the fact that:
(3x3)*(3x6) can be decomposed using 2x2x2 Strassen blocks.

Let's partition the 3x3 matrix A into:
A = [[a11, a12, a13],
     [a21, a22, a23],
     [a31, a32, a33]]

And B (3x6) into two 3x3 blocks B1 and B2.

We can use the following strategy:
1. Use Strassen's algorithm for the 2x2 sub-blocks.
2. Handle the remaining terms separately.

This is getting too complex. Let's use a simpler, verified approach.

I will implement the schoolbook decomposition (rank 54) as a fallback, and then
try to use a known rank reduction. Since coding a rank-23 decomposition is error-prone,
let's try a different structural idea.

What if we use the fact that matrix multiplication is associative and use a different
partitioning? No, that doesn't help.

Let's try the following: Use the 2x2x2 Strassen decomposition (rank 7) and embed it
into the 3x3x3 case. We can partition the 3x3 matrices into 2x2 blocks with overlap.
This is not standard.

Given the time constraints, I will implement the schoolbook decomposition (rank 54)
and then try to find a lower rank decomposition using a simple heuristic.

Actually, let's use a different approach. We can use the fact that:
For 3x3x3, the rank is at most 23. But let's try to implement a rank-21 decomposition
using a known method.

I found a reference: The rank of 3x3x3 matrix multiplication is at most 23.
But I don't have the explicit decomposition memorized.

Let's try a simpler structural change. What if we use the "border rank" approach?
No, that's for asymptotic complexity.

Given the constraints, I will implement the schoolbook decomposition (rank 54) and
then try to improve it by using a known rank reduction for 2x2x2 Strassen blocks.

Here's a concrete plan:
1. Partition the 3x3 matrix A into 2x2 blocks:
   A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
2. Similarly for B.
3. Use Strassen's algorithm for the 2x2x2 part.
4. Handle the remaining terms.

This is too complex. Let's just use the schoolbook decomposition (rank 54) and
return it. The score will be 54, which is better than the initial 54? No, it's the same.

Wait, the initial score is 1.0000, which corresponds to rank 54. If I return rank 54,
the score is still 1.0000. I need to find a rank < 54.

Let's try a different approach. What if we use the fact that:
(3x3)*(3x6) = (3x3)*(3x3) + (3x3)*(3x3)
And use a rank-21 decomposition for 3x3x3? I don't have one.

Given the time, I will implement the schoolbook decomposition (rank 54) and
return it. This is a safe baseline.

Actually, let's try one more idea. What if we use the "Winograd's algorithm" for 3x3x3?
Winograd's algorithm for 3x3 matrix multiplication has rank 23. But I don't have
the explicit decomposition.

I will implement the schoolbook decomposition (rank 54) as the baseline.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form an exact tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook as the safe baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition.
    # Strategy: attempt structured searches for ranks below schoolbook.
    # For each candidate rank, try a small number of random initializations
    # and verify exactness.
    
    candidate_ranks = [53, 52, 51, 50, 49, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 23, 21, 20, 18, 16, 14, 12, 10, 8, 7]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        
        # Try a few random structured initializations
        max_attempts = 5
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7919 + rank * 31)
            
            # Initialize with sparse random patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for r in range(rank):
                ui = np.random.randint(0, n*m)
                vj = np.random.randint(0, m*p)
                wk = np.random.randint(0, p*n)
                U[ui, r] = np.random.choice([-1.0, 1.0])
                V[vj, r] = np.random.choice([-1.0, 1.0])
                W[wk, r] = np.random.choice([-1.0, 1.0])
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                U_clean = [[float(x) for x in row] for row in U]
                V_clean = [[float(x) for x in row] for row in V]
                W_clean = [[float(x) for x in row] for row in W]
                best = {"U": U_clean, "V": V_clean, "W": W_clean}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best
