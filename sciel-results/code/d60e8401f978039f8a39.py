"""
Rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.

Approach:
  - Use the known lower bound for matrix multiplication tensor rank:
        R(n, m, p) >= n*m + m*p + p*n - n - m - p + 1
    For (3,5,5):  15 + 25 + 15 - 3 - 5 - 5 + 1 = 43.
  - Use the classical upper bound via "sum of two smaller multiplications":
        R(3,5,5) <= R(2,5,5) + R(1,5,5)
                  = (2*5 + 5*5 + 5*2 - 2 - 5 - 5 + 1) + (1*5 + 5*5 + 5*1 - 1 - 5 - 5 + 1)
                  = (10 + 25 + 10 - 12) + (5 + 25 + 5 - 10)
                  = 33 + 25 = 58.
  - But a better upper bound comes from the "block diagonal" embedding:
        R(3,5,5) <= R(1,5,5) * 3 = 25 * 3 = 75  (schoolbook).
  - The true rank is not known to be exactly 43 or 58, but we can achieve a
    rank between the lower and upper bounds by using a known construction.

  However, the problem statement says "score is the rank" and "lower is better".
  The seed program uses rank 75 (schoolbook). We need to find a valid
  decomposition with rank < 75.

  Key insight: We can use the fact that matrix multiplication is bilinear.
  The tensor for 3x5*5x5 has 3*5*5 = 75 components. The minimum possible
  rank is at least 43 (as computed above). But we can construct a valid
  decomposition with rank equal to the upper bound from known results.

  In fact, for multiplying an (n x m) matrix by an (m x p) matrix, there
  exists a decomposition of rank at most n*m*p (schoolbook). We can improve
  on this by exploiting the structure.

  A simpler and more direct approach: use the "block" decomposition.
  Write A (3x5) as a column vector of 3 row-vectors a_i (each 1x5).
  Write B (5x5) as a column vector of 5 row-vectors b_j (each 1x5).
  Then (AB)_{ik} = sum_j a_{i,j} b_{j,k}.

  This is exactly the schoolbook formula with rank 75.

  To reduce the rank, we need a non-trivial bilinear algorithm. The simplest
  non-trivial improvement for small dimensions is to use the fact that some
  products can be shared. However, for general matrices (no sparsity), the
  only known improvements over schoolbook come from algorithms like Strassen
  (for 2x2x2) or more sophisticated constructions.

  Given the constraints (exact, small dimensions, 120s time limit), the most
  practical improvement is to use the lower bound 43 as a target and verify
  if a decomposition of rank 43 exists. But constructing such a decomposition
  explicitly is non-trivial.

  Alternative: Use the "rectangular" Strassen-like construction. For 3x5*5x5,
  we can pad to 4x8*8x8 or similar, but this may not help.

  The safest and most reliable improvement: use the known upper bound of 58
  from the sum decomposition R(2,5,5) + R(1,5,5). We can construct this:
    - First 2 rows of A multiplied by B: use a 2x5*5x5 algorithm of rank 33.
    - Last 1 row of A multiplied by B: use a 1x5*5x5 algorithm of rank 25.
    Total rank: 33 + 25 = 58.

  For the 1x5*5x5 case, the rank is 25 (schoolbook for 1x5*5x5).
  For the 2x5*5x5 case, the rank is 33 (using the formula n*m + m*p + p*n - n - m - p + 1
  for n=2, m=5, p=5: 10 + 25 + 10 - 2 - 5 - 5 + 1 = 34, not 33).

  Let me recompute: 10 + 25 + 10 = 45; 45 - 2 - 5 - 5 + 1 = 34.
  So R(2,5,5) <= 34, R(1,5,5) <= 25, total <= 59.

  This is still an improvement over 75.

  Let's construct the 2x5*5x5 decomposition of rank 34 and the 1x5*5x5
  decomposition of rank 25, and combine them.

  For 1x5*5x5 (rank 25):
    A is 1x5, B is 5x5, result is 1x5.
    (AB)_k = sum_j A_j B_{j,k}
    This is a standard dot product of a 1x5 vector with each column of B.
    Rank 25: for each (j,k), one term.

  For 2x5*5x5 (rank 34):
    A is 2x5, B is 5x5, result is 2x5.
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    The formula n*m + m*p + p*n - n - m - p + 1 = 34 gives a lower bound,
    and it is achievable. This is a known result for small matrices.

  However, constructing the explicit rank-34 decomposition for 2x5*5x5 is
  complex. Given the time constraints, I will implement the rank-59
  decomposition (34 + 25) as it is a clear improvement over 75 and
  constructible.

  Actually, let me use a simpler and more reliable approach: the "block
  diagonal" method where we treat the 3x5 matrix as three 1x5 rows.
  Each 1x5*5x5 product has rank 25, so three of them give rank 75 (schoolbook).

  To get rank 59, I need the 2x5*5x5 rank-34 decomposition. Since I cannot
  easily construct it, I will use a different strategy: use the rank-43
  lower bound and hope that a valid decomposition exists at that rank.

  Given the complexity, I will output a valid decomposition with rank 59
  using the sum of 2x5*5x5 (rank 34) and 1x5*5x5 (rank 25). But since
  constructing the rank-34 part is non-trivial, I will use the schoolbook
  for both parts, giving rank 25*2 + 25 = 75 again.

  The most honest and correct approach: output the schoolbook decomposition
  with rank 75, as it is the only one I can guarantee is correct and valid.
"""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
