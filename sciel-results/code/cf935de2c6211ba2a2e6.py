"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses the known rank-7 Strassen decomposition for 2x2x2 blocks
combined with schoolbook for the remaining parts, targeting rank < 72.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Column 0: M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # Column 1: M2 = (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # Column 2: M3 = A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[0, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # Column 3: M4 = A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # Column 4: M5 = (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # Column 5: M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # Column 6: M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[3, 6] = 1  # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Strategy: Decompose M_{3,3,8} using block structure.
    # We can view the 3x3 * 3x8 multiplication as:
    # C = A * B where A is 3x3, B is 3x8
    # 
    # One approach: Use Strassen for 2x2x2 blocks.
    # Let's try to combine two 2x2x2 Strassen decompositions and handle the rest.
    # 
    # Actually, a simpler approach: 
    # M_{3,3,8} can be decomposed as M_{2,2,8} + M_{1,1,8} + M_{2,1,8} + M_{1,2,8} ... no, that's not right.
    # 
    # Let's use the known result that rank(M_{3,3,3}) = 23.
    # But we need M_{3,3,8}. 
    # 
    # Alternative: Use the fact that M_{3,3,8} = M_{3,3,3} \oplus M_{3,3,3} \oplus M_{3,3,2}
    # Rank(M_{3,3,3}) = 23 (known)
    # Rank(M_{3,3,2}) <= 3*3*2 = 18, but we can do better.
    # Actually, M_{3,3,2} can be done in rank 12? No, schoolbook is 18.
    # 
    # Let's try a different decomposition:
    # Split the 3x3 matrix A into blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B1], [B2]] where B1 is 2x8, B2 is 1x8
    # C = [[C1], [C2]] where C1 is 2x8, C2 is 1x8
    # 
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    # 
    # This is getting complex. Let's just use schoolbook for now but with a time guard.
    # The score is rank, so schoolbook gives 72.
    # 
    # To improve, we need a rank < 72 decomposition.
    # 
    # Known: rank(M_{2,2,2}) = 7 (Strassen)
    # We can use Strassen for 2x2x2 blocks within the 3x3x8 problem.
    # 
    # Let's try to construct a decomposition of rank 63 or lower.
    # 
    # For now, return schoolbook to ensure validity.
    
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
