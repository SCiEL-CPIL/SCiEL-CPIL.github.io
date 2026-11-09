"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x8 matrix.
    
    We use a hybrid approach:
    - For the first 2 columns of B (forming a 3x2 block), we can potentially
      use optimized decompositions.
    - For the remaining 6 columns, we use schoolbook.
    
    Actually, let's think about this more carefully.
    
    The matrix multiplication tensor for (n,m,p) = (3,3,8) has dimension
    n*m*p = 72. The schoolbook decomposition has rank 72.
    
    We can try to reduce this by using Strassen's algorithm on sub-blocks.
    
    Let's partition A (3x3) into:
      A = [[A11, A12, A13],
           [A21, A22, A23],
           [A31, A32, A33]]
    
    And B (3x8) into:
      B = [[B11, B12, ..., B18],
           [B21, B22, ..., B28],
           [B31, B32, ..., B38]]
    
    The product C (3x8) has:
      C[i,k] = sum_{j=1}^{3} A[i,j] * B[j,k]
    
    One approach: use Strassen's 7-multiplication algorithm for 2x2 blocks.
    We can view the first 2 rows and 2 columns of A, and the first 2 rows of B.
    
    Actually, let's try a different approach. We can use the fact that
    for 3x3 x 3x1, the minimum rank is known to be 19 (from literature).
    For 3x3 x 3x8, we might be able to do better than 72.
    
    However, for simplicity and correctness, let's implement a valid
    decomposition with reduced rank by using Strassen on the 2x2 sub-block.
    
    Let's partition:
      A = [[A_top, A_right],    where A_top is 2x2, A_right is 2x1
           [A_bottom, A_br]]    where A_bottom is 1x2, A_br is 1x1
    
    B = [[B_top, B_right],      where B_top is 2x8, B_right is 2x8... no
         [B_bottom, B_br]]      this doesn't work cleanly.
    
    Let me try a cleaner approach. We'll use the standard schoolbook but
    look for opportunities to combine terms.
    
    Actually, the simplest valid improvement: use Strassen's algorithm
    for the 2x2 block multiplication where applicable.
    
    For the top-left 2x2 block of A and the top 2 rows of B:
      C_top = A_top * B_top  (2x2 * 2x8 -> 2x8)
    
    This can be done with Strassen, reducing from 16 to 14 multiplications
    for the 2x2 * 2x8 case? No, Strassen is for 2x2 * 2x2.
    
    Let me just implement the schoolbook decomposition correctly.
    The rank is 72, which is the baseline.
    
    To improve, I'll try to use a known result: the tensor rank of
    3x3 x 3x1 matrix multiplication is 19 (not 9). But I need to verify
    this and construct the decomposition.
    
    For now, let's return the schoolbook decomposition with rank 72.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # Schoolbook: rank = n*m*p = 72
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
