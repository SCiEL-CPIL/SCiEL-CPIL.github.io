"""Program for matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    # Use block decomposition with Strassen for 2x2x2 blocks
    # A is 3x4, B is 4x5, C is 3x5
    # Split into blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 2x2, B22 is 2x2
    # Wait, let me think more carefully.
    # A is 3x4: rows 0,1,2; cols 0,1,2,3
    # B is 4x5: rows 0,1,2,3; cols 0,1,2,3,4
    # C is 3x5: rows 0,1,2; cols 0,1,2,3,4
    
    # Let's split A into top 2 rows and bottom 1 row, and left 2 cols and right 2 cols
    # A_top_left: 2x2 (rows 0-1, cols 0-1)
    # A_top_right: 2x2 (rows 0-1, cols 2-3)
    # A_bottom_left: 1x2 (row 2, cols 0-1)
    # A_bottom_right: 1x2 (row 2, cols 2-3)
    
    # B: rows 0-3, cols 0-4
    # Split B into top 2 rows and bottom 2 rows, and left 2 cols and right 3 cols? No...
    # Let's split B into top 2 rows (0-1) and bottom 2 rows (2-3), and left 2 cols (0-1) and right 3 cols (2-4)?
    # That doesn't work well for Strassen.
    
    # Actually, let's just use the trivial decomposition for now since implementing Strassen
    # correctly for mixed block sizes is complex and error-prone.
    # The trivial rank is 3*4*5 = 60.
    
    # Let me try a different approach: use the known result that rank of 3x4x5 <= 56
    # by using Strassen for the 2x2x2 part.
    
    # For simplicity and correctness, let's just return the trivial decomposition.
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
