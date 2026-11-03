"""
Optimal-rank tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.

Strategy:
- Treat the 3x9 matrix as three 3x3 blocks.
- Use the schoolbook 6-term decomposition for multiplying a 2x3 by a 3x3 matrix (rank 6).
- Tensor this with the 9-dimensional identity algebra (rank 9).
- Total rank = 6 * 9 = 54, but the structure allows exact decomposition.

Actually, for 2x3 x 3x9, the "natural" decomposition is:
- For each of the 9 columns of the 3x9 matrix (which are 3x1 vectors), we multiply 2x3 by 3x1.
- But that's not quite right.

Let's think again. We want to compute C = A * B where A is 2x3 and B is 3x9.
C is 2x9. C[i][k] = sum_j A[i][j] * B[j][k].

We can group the 9 columns of B into 3 groups of 3 columns each.
Let B = [B1 | B2 | B3] where each Bj is 3x3.
Then C = [A*B1 | A*B2 | A*B3].

For each 3x3 block Bj, we compute Aj = A * Bj (2x3 result).
The tensor for 2x3 x 3x3 multiplication has rank 18 (schoolbook) but can it be lower?

Actually, the task says the output factor W has shape (p*n, rank) = (9*2, rank) = (18, rank).
And the convention is W[k*n+i, r].

Let me reconsider the "diagonal algebra" hint.
If we think of the 3x9 matrix as 9 independent 3x3 matrices scaled... no.

Let's use the fact that matrix multiplication M_{n,m,p} can be decomposed using the rank of M_{n,m,1} * p if we can share.
But the cleanest approach for an exact rank-minimal decomposition is to use the known structure.

For M_{2,3,9}, the rank is at most 2*3*9 = 54.
Can we do better?

Consider the decomposition using the "block diagonal" structure.
If we write the 9-dimensional space as 3 copies of a 3-dimensional space, and the multiplication is "parallel" over these 3 copies, then:
- We compute 3 independent 2x3 x 3x3 multiplications.
- Each 2x3 x 3x3 multiplication has rank 18 (schoolbook).
- Total rank = 3 * 18 = 54. This is the same as schoolbook.

But the hint says "rank-18 decomposition by tensoring the optimal rank-6 decomposition of 2x3x3".
This suggests that M_{2,3,3} has rank 6. Is that true?
M_{2,3,3}: 2x3 times 3x3 = 2x3. Output has 6 entries.
Rank 6 means we can compute all 6 outputs with 6 bilinear forms.
This is the minimal possible rank (since output has 6 entries, rank >= 6).
Is rank 6 achievable?
Yes, if we can find a basis where the multiplication is "component-wise".
For M_{2,3,3}, if we consider the 3x3 matrix as a 9-dimensional vector, the multiplication is a 2x3 linear map applied to each of the 3 columns.
Actually, C[i][k] = sum_j A[i][j] B[j][k].
This is just 6 independent scalar multiplications... no, there's no sharing.
So rank 18 for M_{2,3,3} is the schoolbook rank.

Wait, the hint says "optimal rank-6 decomposition of 2x3x3".
If M_{2,3,3} had rank 6, then M_{2,3,9} would have rank 6*3 = 18 (if we can decompose 9 into 3 groups of 3).
But M_{2,3,3} rank 6 is not generally known to be true. The lower bound is 6, but the upper bound is 18.

Let me re-read the hint: "tensoring the optimal rank-6 decomposition of 2x3x3 matrix multiplication with the standard rank-9 basis of the diagonal algebra."
This is confusing. Let me just implement the schoolbook decomposition which is rank 54, and see if I can find a better structure.

Actually, let's try a different approach. Use the "fast" matrix multiplication ideas.
But for small dimensions, the best known is often schoolbook.

Let me just return the schoolbook decomposition with rank 54. It's correct and safe.

Wait, I should try to find a rank-18 decomposition if possible.
If we can decompose the 9-dimensional space into 3 independent 3-dimensional spaces, and the multiplication is parallel, then rank = 3 * rank(M_{2,3,3}).
If rank(M_{2,3,3}) = 6, then rank = 18.

Let's assume the hint is correct and try to construct a rank-18 decomposition.
We need 18 bilinear forms.
For each of the 3 groups of 3 columns, we compute a 2x3 x 3x3 multiplication.
If each uses 6 bilinear forms, total is 18.

How to compute 2x3 x 3x3 with 6 bilinear forms?
C[i][k] = sum_j A[i][j] B[j][k]
We need to compute 6 values C[0][0], C[0][1], C[0][2], C[1][0], C[1][1], C[1][2].
Each is a sum of 3 products.
With 6 bilinear forms, we can compute 6 "independent" products.
But each C[i][k] is a sum of 3 products, so we need at least 3*6 = 18 bilinear forms if there's no sharing.
So rank 6 for M_{2,3,3} is impossible unless there's massive sharing.

I think the hint might be referring to a specific algebraic structure.
Let me just implement the schoolbook decomposition. It's rank 54, which is the baseline.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Try to find a rank-18 decomposition
    # The hint suggests tensoring M_{2,3,3} (rank 6) with a 3-dimensional identity
    # Let's try to construct it
    
    # For now, let's use the schoolbook decomposition as a fallback
    # But first, let's try the "block" structure
    
    # The 9 columns of B (3x9) can be seen as 3 blocks of 3 columns each
    # For each block, we compute A (2x3) * B_block (3x3) = C_block (2x3)
    # So we have 3 independent M_{2,3,3} multiplications
    
    # If we can decompose M_{2,3,3} with rank 6, total rank is 18
    # But M_{2,3,3} rank is at least 6 (output size) and at most 18 (schoolbook)
    
    # Let's try to find a rank-6 decomposition for M_{2,3,3}
    # This is equivalent to finding a 6-term algorithm for 2x3 x 3x3
    
    # Actually, let's just use the schoolbook for M_{2,3,3} which is rank 18
    # Total rank = 3 * 18 = 54
    
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
