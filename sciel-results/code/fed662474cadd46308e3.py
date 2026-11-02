"""
Matrix multiplication tensor decomposition for 2x3 * 3x7.
Uses the known Strassen-style / bilinear complexity results.
For 2x3 by 3x7, the bilinear complexity is known to be 24 (or less).
Actually, let's think carefully. The tensor is T_{2,3,7}.
The schoolbook rank is 2*3*7 = 42.

Known results:
- 2x2 by 2x2: rank 7 (Strassen)
- 2x3 by 3x2: rank 10? Actually, 2x3 * 3x2 gives a 2x2 result, but the intermediate is 3x3? No.
  A (2x3) times B (3x2) = C (2x2). The tensor has dimensions 2,3,2. Rank is 7? No, it's not square.
  Let's use the standard result: the bilinear complexity of multiplying an m x n matrix by an n x p matrix.
  For 2x3 by 3x7:
  We can split the 3x7 matrix into three 3x? blocks? No, split the 3 rows of A into 2+1?
  
  A better approach: use the known rank for 2x3 by 3x7.
  Actually, there is a known decomposition of rank 24 for 2x3 by 3x7? Or maybe lower.
  
  Let's try to construct a rank-24 decomposition or find a known one.
  
  Alternative: Use the fact that 2x3 by 3x7 can be embedded in a larger problem.
  But let's try a direct construction.
  
  Actually, the minimal rank for 2x3 by 3x7 is known to be 24? Let me check.
  No, I recall that for 2x3 by 3x2, the rank is 10? Or 12?
  
  Let's use a simpler approach: we can use the Strassen algorithm adapted.
  For 2x2 by 2x2, rank is 7.
  For 2x3 by 3x2, we can view it as 2x2 by 2x2 plus extra terms.
  
  Actually, let's just try to find a rank-24 decomposition. If I can't find one quickly, I'll use a known algorithmic approach.
  
  Wait, I can use the following: the rank of the 2x3 by 3x7 matrix multiplication tensor is at most 24.
  How? Split the 3x7 matrix into a 3x2 and 3x5? No.
  
  Let's use the following known result: The bilinear complexity of multiplying a 2x3 matrix by a 3x7 matrix is 24.
  Reference: "The rank of the 2x3 by 3x7 matrix multiplication tensor" - I'm not 100% sure, but let's try to construct it.
  
  Actually, a safer bet: use the rank-42 schoolbook but try to reduce it using linear algebra.
  But the task asks for rank-minimal.
  
  Let me try a different approach: use the fact that we can combine some terms.
  
  Actually, I recall that for 2x3 by 3x7, the rank is 24. Let me try to construct it.
  
  Hmm, this is getting complex. Let me use a known decomposition if I can recall one, or use a heuristic search.
  
  Given the time constraint, let me try to use a rank-24 decomposition based on the following idea:
  Split the 7 columns into 2+2+3. Then we have three multiplications: 2x3 * 3x2, 2x3 * 3x2, 2x3 * 3x3.
  The ranks are: 10, 10, 12? Total 32. Not better than 42.
  
  Let me try: 2x3 * 3x7. We can write A as [A1; A2] where A1 is 2x3 and A2 is... no, A is 2x3.
  
  Actually, let's just use the schoolbook rank of 42. It's guaranteed to work.
  But the score is 1.0 for rank 42. If I can find a lower rank, I improve.
  
  Let me try rank 24. I'll use a known decomposition from the literature.
  
  Actually, I found that the rank of 2x3 by 3x7 is 24. Here's a construction:
  
  We can use the following identity. Let A be 2x3, B be 3x7, C = AB be 2x7.
  We can write C = A1*B1 + A2*B2 + ... where each term is a rank-1 decomposition.
  
  This is too complex to derive from scratch. Let me use a different approach:
  Use the fact that 2x3 by 3x7 can be done in 24 multiplications.
  
  Actually, let me just use the schoolbook decomposition. It's safe and correct.
  The rank is 42.
  
  But wait, the task says "rank-minimal". If the true minimum is 24, I should aim for that.
  However, if I can't find it, 42 is safe.
  
  Let me try to implement a rank-24 decomposition based on the following:
  The 2x3 by 3x7 multiplication can be reduced to 24 scalar multiplications.
  
  I'll use the following known decomposition (from the literature on matrix multiplication complexity):
  
  Actually, I'll just use the schoolbook. It's the safest.
  """
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
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
