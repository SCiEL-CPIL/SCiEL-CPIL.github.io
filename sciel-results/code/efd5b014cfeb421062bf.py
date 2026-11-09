"""
Kronecker-product construction for 3x3 x 3x8 matrix multiplication.

Strategy:
  The multiplication tensor for (n,m,p) = (3,3,8) has entries T[i,j,k] such that
  C[i,k] = sum_j A[i,j] B[j,k].

  We can view this as a Kronecker product of two smaller tensors:
  - A 3x3 x 3x1 tensor (matrix-vector multiplication): rank 9 (schoolbook)
  - A 1x1 x 1x8 tensor (identity/scaling): rank 8

  But that gives 9*8 = 72, which is the same as schoolbook.

  A better approach: use the fact that the 3x3 x 3x3 multiplication tensor has
  rank at most 23 (Strassen). For 3x3 x 3x8, we can decompose the 8 columns
  into blocks. However, the cleanest improvement comes from noting that:

  C[i,k] = sum_j A[i,j] B[j,k]

  This is equivalent to computing, for each k, the product of A (3x3) with
  the k-th column of B (3x1). The bilinear map for one column has rank 9
  (schoolbook). But we can share computations across columns if we use
  a smarter decomposition.

  Actually, the key insight: the tensor for (3,3,8) multiplication is the
  Kronecker product of the (3,3,1) tensor and the (1,1,8) tensor.
  The (3,3,1) tensor has rank 9 (it's just the 9 scalar multiplications for
  matrix-vector product). The (1,1,8) tensor has rank 8 (identity).
  Kronecker product rank is 9*8 = 72.

  To improve, we need a non-Kronecker approach. Let's use the known fact that
  the rank of (n,m,p) multiplication is at most n*m*p, and for specific cases
  we can do better.

  For (3,3,8): one approach is to note that 8 = 3 + 3 + 2, and use:
  - Rank of (3,3,3) is at most 23 (Strassen)
  - Rank of (3,3,3) is at most 23
  - Rank of (3,3,2) is at most 18 (schoolbook)
  Total: 23 + 23 + 18 = 64

  This is an improvement over 72. Let's implement the Strassen decomposition
  for the 3x3 x 3x3 blocks and schoolbook for the 3x3 x 3x2 block.

  Strassen's algorithm for 3x3 x 3x3:
  We partition the 3x3 matrices into 2x2 blocks with overlapping or use
  the known rank-23 decomposition.

  Actually, the standard Strassen algorithm is for 2x2 x 2x2 with rank 7.
  For 3x3, we can use a different approach. The known upper bound for 3x3x3
  is 23, achieved by specific decompositions.

  For simplicity and correctness, let's use a decomposition that achieves
  rank 54 by using a Kronecker product of (3,3,3) with rank 23 and (1,1,8/3)...
  no, that doesn't work cleanly.

  Let me use a direct approach: the rank of (3,3,8) multiplication is at most
  the sum of ranks of (3,3,k) for k=1 to 8, but that's 8*9=72.

  A better bound: the rank of (n,m,p) multiplication is at most n*m*p, but
  also at most n*p*m (trivially). For (3,3,8), we can use the fact that the
  tensor is a "slice" of a larger tensor.

  Given the complexity of implementing a verified rank-23 decomposition for 3x3x3,
  and the risk of errors, I'll implement a correct decomposition with rank 54
  using a block approach:

  Split the 8 columns into 3+3+2. For each block of size b, the rank of
  (3,3,b) multiplication is 9*b (schoolbook). So:
  - (3,3,3): rank 27
  - (3,3,3): rank 27  
  - (3,3,2): rank 18
  Total: 72. No improvement.

  To get an improvement, I need to use a non-schoolbook decomposition for at
  least one of the blocks. The Strassen decomposition for (2,2,2) has rank 7
  instead of 8. If I can embed this, I might save some rank.

  For (3,3,3), I can use a decomposition that achieves rank 23. Here's a known
  construction:

  Actually, let me just output the schoolbook decomposition which is guaranteed
  correct. The task asks for a rank-minimal decomposition, but without a verified
  better decomposition, I'll stick with the safe option.

  Wait - I can use a simple improvement: the rank of (3,3,8) is at most 3*3*8=72,
  but also can be bounded by considering that the multiplication is linear in
  the columns of B. Each column requires rank 9, but if we share some computations...

  For now, I'll output the schoolbook decomposition.
"""

DIMENSIONS = [3, 3, 8]

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
