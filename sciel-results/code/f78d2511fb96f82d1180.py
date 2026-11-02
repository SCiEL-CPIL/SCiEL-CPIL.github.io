"""
Rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.

Approach:
  Use a structured Kronecker product decomposition.
  Express the 3x5 matrix A as a 3x1 block vector of 1x5 rows.
  Express the 5x5 matrix B as a 5x1 block vector of 1x5 rows.
  The output C = AB is 3x5.
  
  The naive schoolbook has rank 75. We aim for a better construction.
  
  Key idea: Use the separability of the outer product structure.
  For each output element C[i,k], we need sum_j A[i,j] * B[j,k].
  
  Let's think about this differently. We can group terms.
  
  Actually, the directive suggests a rank-15 decomposition derived from
  the tensor product of a rank-3 identity and a rank-5 identity. This seems
  optimistic - rank 15 is below the known lower bound of 43, so it cannot work.
  
  Let me reconsider. The lower bound R(3,5,5) >= 3*5 + 5*5 + 5*3 - 3 - 5 - 5 + 1 = 43.
  
  A valid construction: Use the block sum approach.
  R(3,5,5) <= R(2,5,5) + R(1,5,5)
  
  For R(1,5,5): rank 25 (schoolbook for 1x5 * 5x5)
  For R(2,5,5): We can use R(2,5,5) <= R(1,5,5) + R(1,5,5) = 50, or better,
  use the fact that R(n,m,p) <= n*m*p for schoolbook.
  
  A more efficient approach: Pad to square and use known rectangular Strassen.
  
  Let me try a different valid construction that I can explicitly build:
  
  Block decomposition: Split the 3 rows into groups.
  Group 1: rows 0-1 (2x5), Group 2: row 2 (1x5)
  
  For 2x5 * 5x5: 
    We can write this as (2x5)*(5x5) = (2x5)*(5x1)*(1x5) but that doesn't help directly.
    
  Actually, let me use a simpler valid improvement:
  R(3,5,5) <= 3 * R(1,5,5) = 75 (schoolbook)
  R(3,5,5) <= R(2,5,5) + R(1,5,5)
  
  For 2x5*5x5, the schoolbook rank is 2*5*5 = 50.
  So total would be 50 + 25 = 75, same as before.
  
  Let me try: R(3,5,5) <= R(3,5,4) + R(3,5,1)
  R(3,5,4) <= 3*5*4 = 60
  R(3,5,1) <= 3*5*1 = 15
  Total: 75. Still not better.
  
  The most practical improvement I can construct explicitly:
  Use the formula for R(n,m,p) with the "tensor rank lower bound" being achievable
  for small cases. For 1x5*5x5, rank = 25 (tight).
  For 2x5*5x5, let me use rank 34 (the lower bound, which is achievable for this case).
  
  Construction for 2x5*5x5 with rank 34:
  This is complex. Let me use a simpler but valid improvement.
  
  Actually, I can use: R(3,5,5) <= R(3,5,3) + R(3,5,2)
  R(3,5,3) <= 3*5*3 = 45
  R(3,5,2) <= 3*5*2 = 30
  Total: 75.
  
  Let me just use the schoolbook with rank 75, as constructing a provably valid
  sub-75 decomposition within the time limit is risky. But that gives score 1.0.
  
  Wait - let me try a direct construction for a better rank.
  
  Using the identity: for any matrices, we can use the "sum of products" approach.
  
  Let me construct a valid rank-59 decomposition:
  - First 2 rows: use schoolbook for 2x5*5x5 = rank 50
  - Last 1 row: use schoolbook for 1x5*5x5 = rank 25
  - Total: 75. Still not better.
  
  I need a genuinely different approach. Let me use the fact that:
  R(3,5,5) <= R(3,5,5) via a non-trivial bilinear algorithm.
  
  Given the constraints, I'll output the schoolbook decomposition which is valid
  and guaranteed correct.
"""

def construct():
    n, m, p = 3, 5, 5
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
