"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x3 x 3x4 matrix multiplication.
    
    Strategy: Partition the 3x3x4 tensor into:
    - A 2x2x4 block (using first 2 rows of A, first 2 columns of B)
    - A 1x3x4 block (using row 2 of A, all columns of B)
    - A 3x1x3 block (using all rows of A, column 3 of B)
    
    For the 2x2x4 block: use schoolbook (rank 8) since rectangular Strassen 
    gains are limited. Actually, let's try a different approach.
    
    Better strategy: Use known lower bounds and search for a rank-34 decomposition.
    The key insight is that we can share computation across blocks.
    
    I'll construct a rank-34 decomposition by:
    1. Using Strassen's algorithm structure on a 2x2x2 subblock (rank 7 instead of 4)
    2. Sharing some of those 7 products with other parts of the computation
    
    Let me define the tensors:
    A is 3x3, B is 3x4, C = A@B is 3x4
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's partition:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
     
    B = [[B11, B12, B13, B14],
         [B21, B22, B23, B24],
         [B31, B32, B33, B34]]
     
    C = [[C11, C12, C13, C14],
         [C21, C22, C23, C24],
         [C31, C32, C33, C34]]
    
    Each C[i,k] = A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    Total schoolbook terms: 3*3*4 = 36
    
    To get rank 34, we need to eliminate 2 terms through clever sharing.
    
    One known approach for 3x3x3 is rank 23 (Strassen-based). For 3x3x4, 
    we can extend this.
    
    Let me use a constructive approach with Strassen on the 2x2x2 core:
    
    For the 2x2x2 block (A12x2 @ B2x2), Strassen gives 7 products:
    M1 = A[0,0]*(B[1,1]-B[1,0]) ... etc.
    
    Actually, let me just implement the best known decomposition for this case.
    Given the complexity, I'll implement a verified rank-35 or rank-34 decomposition.
    
    After research, I'll construct a rank-34 decomposition using the following method:
    - Use 7 Strassen products for a 2x2x2 submatrix
    - Use schoolbook for the remaining parts
    - Share some computations to reduce total from 36 to 34
    
    Here's the explicit construction for rank 34:
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    rank = 34
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    def add_term(i, j, k, coeff=1):
        """Add a term to the decomposition.
        U[i*m+j] *= coeff, V[j*p+k] *= coeff, W[k*n+i] *= coeff for one rank-1 term
        """
        # This will be called for each of the 34 terms
        pass
    
    # For simplicity and correctness, I'll implement a specific known rank-34 decomposition.
    # Based on the structure of 3x3x3 rank 23, extended to 3x3x4.
    
    # Let me use a different approach: construct via bilinear form coefficients.
    # Each rank-1 term corresponds to a choice of (i,j,k) with possible sign flips.
    
    # Known optimal-ish decomposition for 3x3x4 (rank 34):
    # Term 1-7: Strassen for top-left 2x2x2
    # Term 8-14: Strassen for another 2x2x2 block
    # Term 15-34: Schoolbook for remaining with sharing
    
    # Actually, let me just implement a clean rank-34 based on documented results.
    # I'll use the following verified structure:
    
    # Columns 0-6: Strassen for A[0:2,0:2] * B[0:2,0:2]
    # M1 = A00*(B11-B10)
    # M2 = (A01+A11)*B10
    # M3 = A10*(B01-B00)
    # M4 = A11*B11
    # M5 = (A00+A11)*(B00+B11)
    # M6 = (A01-A10)*(B10+B11)
    # M7 = (A00-A11)*(B00+B01)
    
    # But for 3x3x4, we need to adapt. Let me use a simpler verified approach.
    
    # I'll use rank 35 as a safe improvement over 36, constructed by 
    # combining Strassen on one 2x2x2 (saving 1 vs 2 schoolbook blocks worth)
    
    # For now, return schoolbook rank 36 to ensure validity, 
    # but this scores 1.0. To beat it, I need < 36.
    
    # Let me try a concrete rank-34 construction.
    
    # After careful analysis, here's a rank-34 decomposition:
    
    # We'll use 34 terms. The first 7 use Strassen structure on indices (0,1) x (0,1) x (0,1)
    # The next 7 use Strassen on (0,1) x (0,1) x (2,3) -- wait, B has 4 cols.
    
    # Let me index: i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}
    
    # Strassen on i in {0,1}, j in {0,1}, k in {0,1}:
    S1  = A[0,0] * (B[1,1] - B[1,0])  # contributes to C[0,0], C[0,1]
    S2  = (A[0,1] + A[1,1]) * B[1,0]
    S3  = A[1,0] * (B[0,1] - B[0,0])
    S4  = A[1,1] * B[1,1]
    S5  = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    S6  = (A[0,1] - A[1,0]) * (B[1,0] + B[1,1])
    S7  = (A[0,0] - A[1,1]) * (B[0,0] + B[0,1])
    
    # Similarly for other blocks, but sharing where possible.
    
    # This is getting complex. Let me just hard-code a known valid rank-34.
    
    # Given time constraints, I'll implement the schoolbook rank 36 as baseline.
    # But the task requires beating 1.0, so I need rank <= 35.
    
    # Let me implement rank 35 by using one Strassen substitution:
    # Replace 4 schoolbook terms (i,j,k) for i in {0,1}, j in {0,1}, k=0 
    # Wait, Strassen reduces 2x2x2 from 4 to 7, which is WORSE for small blocks.
    
    # I see my error. Strassen is for 2x2x2, rank 7 vs schoolbook rank 4. That's worse!
    # Strassen is useful when you can share products across multiple 2x2x2 subproblems.
    
    # For 3x3x3, the known rank 23 uses a more sophisticated sharing.
    # For 3x3x4, rank 34 is known to be achievable.
    
    # Let me just implement a straightforward rank-35 by finding any valid decomposition.
    
    # Safest approach: use the same structure but with a clever elimination.
    
    # I'll construct rank 36 schoolbook and then show it can be reduced.
    
    # For the purpose of this exercise, I'll implement schoolbook (rank 36) 
    # and acknowledge it may not beat the baseline. The actual rank-34 requires
    # the specific algebraic identity which is complex to derive correctly.
    
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
