"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Use Strassen-Winograd recursive approach
    # For 3x3 * 3x8, we can split the 3x8 matrix into a 3x3 and 3x5 block
    # Or better: use the fact that we can apply Strassen to sub-blocks
    
    # Strategy: Split B (3x8) into B1 (3x3) and B2 (3x5)
    # C = A * B1 + A * B2
    # For A*B1: 3x3 * 3x3, use rank-7 decomposition (Strassen)
    # For A*B2: 3x3 * 3x5, use schoolbook (3*3*5 = 45) or further decomposition
    # Total: 7 + 45 = 52
    
    # But we can do better by splitting more cleverly
    # Let's use: B = [B1 | B2 | B3] where B1 is 3x3, B2 is 3x3, B3 is 3x2
    # C = A*B1 + A*B2 + A*B3
    # A*B1: rank 7 (Strassen)
    # A*B2: rank 7 (Strassen)  
    # A*B3: 3x3 * 3x2, schoolbook rank 18
    # Total: 7 + 7 + 18 = 32
    
    # Even better: use recursive Strassen-like decomposition
    # For 3x3 * 3x8, think of it as computing 8 columns of output
    # Each column is a 3x3 * 3x1 multiplication (rank 3 each, total 24)
    # But we can batch: use Strassen on groups of columns
    
    # Let's implement a concrete decomposition:
    # Split the 8 columns into two groups of 4
    # C = A * [B1 | B2] where B1, B2 are 3x4
    # For each 3x3 * 3x4, we can use a rank-12 decomposition (3x4 schoolbook)
    # Or better: use Strassen-like for 3x4
    
    # Actually, let's use a known result: 
    # 3x3 * 3x3 has rank 7 (Strassen)
    # 3x3 * 3x4: we can view as 3x3 * [3x3 | 3x1]
    #   = (rank 7 for 3x3*3x3) + (rank 3 for 3x3*3x1) = 10
    # So 3x3 * 3x8 = 3x3 * [3x3 | 3x3 | 3x2]
    #   = 7 + 7 + (3x3*3x2 schoolbook = 18) = 32
    # Or 3x3 * 3x8 = 3x3 * [3x4 | 3x4]
    #   = 10 + 10 = 20
    
    # Let's try to construct the rank-20 decomposition
    # 3x3 * 3x4 decomposition:
    #   Split B (3x4) into B_left (3x3) and B_right (3x1)
    #   Use Strassen for A*B_left (rank 7)
    #   Use schoolbook for A*B_right (rank 3)
    #   Total rank 10 per 3x4 block
    
    # Strassen 3x3*3x3 decomposition (rank 7):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B01
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M2 - M1 + M5 - M7
    
    # Wait, that's for 2x2. For 3x3, the standard Strassen doesn't directly apply.
    # The rank-7 decomposition for 3x3*3x3 is more complex.
    
    # Let me use a different approach: direct construction
    # For 3x3 * 3x8, let's try to find a decomposition with rank < 72
    
    # Simple improvement: use the fact that we can share computations
    # between the 8 output columns
    
    # Let's try a randomized approach to find a lower rank decomposition
    # by solving a system of equations
    
    # The tensor equation: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if j is the "middle" index
    #   Actually, the multiplication tensor is:
    #   T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
    #   So for each (i,j,k) triple, we need:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 72 equations in rank*r*3 variables (3*r variables per rank term, but with constraints)
    # Actually, each rank term has 3 factors, so for rank r, we have 3*r*(n*m + m*p + p*n) / 3 = r*(n*m + m*p + p*n) variables
    # But the equations are nonlinear (trilinear)
    
    # Let's try a structured decomposition:
    # For each j in [0,m), we have n*p = 24 terms
    # The j-th "slice" is: sum_{i,k} A[i,j] * B[j,k] * C[k,i]
    # This is rank n*p = 24 for each j, total 72
    
    # To reduce rank, we need to share rank terms across different j values
    # This is possible if we use linear combinations
    
    # Let's try: for each pair of j values, see if we can merge their rank terms
    # This is equivalent to finding a low-rank approximation of the multiplication tensor
    
    # Given the complexity, let's return the schoolbook decomposition
    # which is guaranteed correct
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
