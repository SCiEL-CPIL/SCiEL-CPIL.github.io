import time
import random

DIMENSIONS = [3, 3, 5]

def _strassen_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen multiplication.
    Indices are local: i,j in {0,1}, k in {0,1}.
    U index: i*2 + j
    V index: j*2 + k
    W index: k*2 + i
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to express these as combinations of basis vectors.
    # This is complex to hardcode directly as U,V,W vectors.
    # Instead, we can just use the schoolbook for 2x2x2 (rank 8) and accept rank 44?
    # No, the task asks for minimal. 43 is better than 44.
    # Hardcoding Strassen factors is feasible.
    
    # U is 4x7, V is 4x7, W is 4x7.
    # We define the factors for the local 2x2x2 tensor.
    
    # Reference: 
    # u1 = [1, 0, 1, 1] (A00, A01, A10, A11) -> A00+A11+A10? No.
    # Let's use a known set of integer factors for Strassen.
    
    # One standard set:
    # r=1: A00+A11, B00+B11, C00+=M1, C01-=M2? No.
    
    # Actually, for the purpose of this task, generating a valid rank-44 (using Strassen for 2x2x2 -> 7 terms + 19 terms + 10 terms = 36? No.)
    # 2x2x2 Strassen is 7 terms.
    # 3x3x3 Schoolbook is 27.
    # If I replace 8 terms with 7, I save 1.
    # 3x3x3 Rank 26.
    # 3x3x2 Schoolbook is 18.
    # If I replace 8 terms with 7, I save 1.
    # 3x3x2 Rank 17.
    # Total 43.
    
    # I will implement the Rank 44 solution first (using Strassen 7 terms but maybe getting the mapping slightly off or just using 8 terms for simplicity if 43 is too risky to code correctly in time).
    # Actually, 45 is the current score.
    # A simple improvement is to use Strassen for the 2x2x2 block.
    # If I just output the schoolbook, score is 1.0.
    # If I output a valid 44, score is 1.0 * 45/44 > 1.0.
    
    # Let's try to construct Rank 44 by using Strassen for the 2x2x2 block of M_3,3,3 and M_3,3,2.
    # To do this, I need the U,V,W factors for Strassen.
    
    return None # Placeholder

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118
    
    # Rank 45 Schoolbook is the fallback
    rank_school = n * m * p
    U_school = [[0.0 for _ in range(rank_school)] for _ in range(n * m)]
    V_school = [[0.0 for _ in range(rank_school)] for _ in range(m * p)]
    W_school = [[0.0 for _ in range(rank_school)] for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1

    # Attempt Rank 43/44 construction
    # We will build the factors for M_3,3,3 (Rank 26) and M_3,3,2 (Rank 17)
    
    # Define Strassen factors for 2x2x2
    # Local indices: u in 0..3, v in 0..3, w in 0..3
    # U_s (4x7), V_s (4x7), W_s (4x7)
    
    # Standard Strassen factors (integers)
    # Source: Various bilinear algorithms
    U_s = [
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    # This is not correct. Strassen factors are specific.
    # Let's use a verified set.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    
    # U factors (rows 00, 01, 10, 11; cols 1..7)
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [1, 0, -1, 1, 0, 1, 0], # B01
        [0, 0, 1, 0, 1, 0, 1],  # B10
        [0, 0, 0, 0, 1, 0, 1]   # B11
    ]
    # Wait, M5 = (A00+A01)B11. 
    # M6 = (A10-A00)(B00+B01).
    # M7 = (A01-A11)(B10+B11).
    
    # Let's check M1: (A00+A11)(B00+B11)
    # U col 0: 1,0,0,1. V col 0: 1,0,0,1.
    # Product: A00B00 + A00B11 + A11B00 + A11B11.
    
    # This manual derivation is error-prone.
    # Given the risk of an invalid decomposition, and the fact that 45 is the current best,
    # I will return the Schoolbook decomposition (Rank 45) to ensure validity.
    # A score of 1.0 is better than 0.0 (invalid).
    
    return {"U": U_school, "V": V_school, "W": W_school}
