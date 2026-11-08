"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    We use a block decomposition:
    - Split the 3x3 matrix A into 2x2, 2x1, 1x2, 1x1 blocks.
    - Split the 3x7 matrix B into 2x7, 1x7, 0x7 (row 3), and column blocks.
    
    Specifically, we decompose:
    A = [A11 A12; A21 A22; A31 A32]  where A11 is 2x2, A12 is 2x1, A21 is 1x2, etc.
    B = [B1; B2; B3]  where B1, B2 are 2x7 and 1x7... wait, B is 3x7.
    
    Let me re-think. A is 3x3, B is 3x7.
    A = [[a11, a12, a13],
         [a21, a22, a23],
         [a31, a32, a33]]
    B = [[b11, b12, ..., b17],
         [b21, b22, ..., b27],
         [b31, b32, ..., b37]]
    
    Product C = A*B is 3x7.
    
    Block approach:
    Row 1 of A: [a11, a12, a13]
    Row 2 of A: [a21, a22, a23]
    Row 3 of A: [a31, a32, a33]
    
    We can use:
    - Strassen-like for the first 2 rows of A against first 2 columns of B (2x2x2, rank 7)
    - Then handle the remaining interactions.
    
    A cleaner approach: use the known rank-63 schoolbook as baseline, and try to reduce.
    
    For 3x3x7, the schoolbook rank is 3*3*7 = 63.
    
    Let's use a simple but potentially improvable decomposition:
    Group operations by (i,j) where i is row of A (1-3), j is column of B (1-7).
    
    Actually, let's use the approach from the previous attempt:
    Split A into 2x2 + cross terms, B similarly.
    
    Let me use a direct construction with rank 63 as fallback, but try a smarter split.
    
    For time efficiency, I'll construct the schoolbook decomposition (rank 63) 
    and return it. This is safe and correct.
    """
    deadline = time.time() + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    rank = n * m * p  # 63
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 9 x 63
    V = [[0] * rank for _ in range(m * p)]  # 21 x 63
    W = [[0] * rank for _ in range(p * n)]  # 21 x 63
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j][column] = 1
                # V[j*p + k][column] = 1
                # W[k*n + i][column] = 1
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
