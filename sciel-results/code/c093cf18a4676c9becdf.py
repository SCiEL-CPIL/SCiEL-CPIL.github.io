"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-14 tensor decomposition for 2x2 by 2x4 matrix multiplication.
    Based on applying Strassen's algorithm to two pairs of columns.
    Rank 14 is the known minimum rank for M_{2,2,4}.
    """
    n, m, p = DIMENSIONS
    rank = 14
    
    # U shape (n*m, rank) = (4, 14)
    # V shape (m*p, rank) = (8, 14)
    # W shape (p*n, rank) = (8, 14)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_vals, v_vals, w_vals, col):
        for i in range(n * m):
            U[i][col] = u_vals[i]
        for j in range(m * p):
            V[j][col] = v_vals[j]
        for k in range(p * n):
            W[k][col] = w_vals[k]

    # Indices for A (4): 0:00, 1:01, 2:10, 3:11
    # Indices for B (8): 0:00, 1:01, 2:02, 3:03, 4:10, 5:11, 6:12, 7:13
    # Indices for C (8): 0:00, 1:10, 2:01, 3:11, 4:02, 5:12, 6:03, 7:13
    # W index is k*n + i.
    
    # Left Part (Cols 0, 1) - Strassen for 2x2x2
    # 1. (A00+A11)(B00+B11)
    add_term([1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], 0)
    
    # 2. (A10+A11)B00
    add_term([0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0], 1)
    
    # 3. A00(B01-B11)
    add_term([1, 0, 0, 0], [0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], 2)
    
    # 4. A11(B10-B00)
    add_term([0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], 3)
    
    # 5. (A00+A01)B11
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], 4)
    
    # 6. (A10-A00)(B00+B01)
    add_term([-1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], 5)
    
    # 7. (A01-A11)(B10+B11)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], 6)
    
    # Right Part (Cols 2, 3) - Strassen for 2x2x2
    # 8. (A00+A11)(B02+B13)
    add_term([1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1], 7)
    
    # 9. (A10+A11)B02
    add_term([0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, -1], 8)
    
    # 10. A00(B03-B13)
    add_term([1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], 9)
    
    # 11. A11(B12-B02)
    add_term([0, 0, 0, 1], [0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0], 10)
    
    # 12. (A00+A01)B13
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0, 1, 0], 11)
    
    # 13. (A10-A00)(B02+B03)
    add_term([-1, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], 12)
    
    # 14. (A01-A11)(B12+B13)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0], 13)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
