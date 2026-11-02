"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using 8 Strassen 2x2x2 multiplications
    
    U = [[0] * 56 for _ in range(n * m)] # 8 x 56
    V = [[0] * 56 for _ in range(m * p)] # 32 x 56
    W = [[0] * 56 for _ in range(p * n)] # 16 x 56
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We have 8 sub-multiplications.
    # A is 2x4. B is 4x8. C is 2x8.
    # 
    # Sub-multiplication mapping:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 3. A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]  (Wait, B is 4x8, rows 0,1 are top)
    # 4. A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    # 5. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]  (A_R, B_BL)
    # 6. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # 7. A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # 8. A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
    
    # Wait, C is 2x8.
    # C[0:2, 0:2] is sum of contributions from 1 and 5.
    # C[0:2, 2:4] is sum of contributions from 2 and 6.
    # C[0:2, 4:6] is sum of contributions from 3 and 7.
    # C[0:2, 6:8] is sum of contributions from 4 and 8.
    
    # Let's define the 8 sub-problems as (A_row, A_col, B_row, B_col, C_row, C_col)
    # A_row, A_col are start indices in 2x4 A.
    # B_row, B_col are start indices in 4x8 B.
    # C_row, C_col are start indices in 2x8 C.
    
    sub_problems = [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 0, 0, 4, 0, 4),
        (0, 0, 0, 6, 0, 6),
        (0, 2, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 2),
        (0, 2, 2, 4, 0, 4),
        (0, 2, 2, 6, 0, 6)
    ]
    
    col = 0
    for sp in sub_problems:
        ar, ac, br, bc, cr, cc = sp
        
        # Add 7 Strassen columns for this sub-problem
        
        # M1
        U[ar * 4 + ac][col] += 1
        U[(ar+1) * 4 + (ac+1)][col] += 1
        V[br * 8 + bc][col] += 1
        V[(br+1) * 8 + (bc+1)][col] += 1
        W[(cc) * 2 + (cr)][col] += 1
        W[(cc+1) * 2 + (cr+1)][col] += 1
        col += 1
        
        # M2
        U[(ar+1) * 4 + ac][col] += 1
        U[(ar+1) * 4 + (ac+1)][col] += 1
        V[br * 8 + bc][col] += 1
        W[(cc+1) * 2 + (cr+1)][col] += 1
        col += 1
        
        # M3
        U[ar * 4 + ac][col] += 1
        V[(br) * 8 + (bc+1)][col] += 1
        V[(br+1) * 8 + (bc+1)][col] -= 1
        W[(cc+1) * 2 + cr][col] += 1
        col += 1
        
        # M4
        U[(ar+1) * 4 + (ac+1)][col] += 1
        V[(br+1) * 8 + bc][col] += 1
        V[br * 8 + bc][col] -= 1
        W[(cc) * 2 + cr][col] += 1
        W[(cc+1) * 2 + (cr+1)][col] += 1
        col += 1
        
        # M5
        U[ar * 4 + ac][col] += 1
        U[ar * 4 + (ac+1)][col] += 1
        V[(br+1) * 8 + (bc+1)][col] += 1
        W[(cc) * 2 + cr][col] -= 1
        W[(cc+1) * 2 + cr][col] += 1
        col += 1
        
        # M6
        U[(ar+1) * 4 + ac][col] += 1
        U[ar * 4 + ac][col] -= 1
        V[br * 8 + bc][col] += 1
        V[br * 8 + (bc+1)][col] += 1
        W[(cc+1) * 2 + (cr+1)][col] += 1
        col += 1
        
        # M7
        U[ar * 4 + (ac+1)][col] += 1
        U[(ar+1) * 4 + (ac+1)][col] -= 1
        V[(br+1) * 8 + bc][col] += 1
        V[(br+1) * 8 + (bc+1)][col] += 1
        W[(cc) * 2 + cr][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
